"""
Training Script for Brain Stroke Prediction — Hybrid Model
Production-grade 2-phase fine-tuning.

Strategy:
  Phase 1 (Epochs 1-10):  Freeze backbone → train only heads at LR=1e-3
  Phase 2 (Epochs 11-50): Unfreeze backbone → discriminative LR
                          backbone=1e-5, heads=1e-4 + linear warmup
                          BatchNorm stays FROZEN to preserve ImageNet stats

Features:
  ✓ 2-phase training (freeze → unfreeze)
  ✓ Discriminative learning rates
  ✓ Linear warmup for Phase 2
  ✓ Backbone BatchNorm always in eval mode
  ✓ Mixed precision (BF16 on H200, FP16 fallback)
  ✓ Cosine annealing scheduler with warm restarts
  ✓ Class-weighted cross-entropy + label smoothing
  ✓ Gradient clipping (max_norm=1.0)
  ✓ Early stopping by validation AUC-ROC
  ✓ Reloads best Phase 1 checkpoint before Phase 2
  ✓ Saves best + last model checkpoints
  ✓ Trains image-only model for Grad-CAM

NOTE: torch.compile() is intentionally NOT used.
      It causes validation divergence on medical imaging fine-tuning
      tasks due to graph capture issues with frozen/unfrozen BN layers.
"""
import os
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)
from tqdm import tqdm

from config import (
    DEVICE, PHASE1_EPOCHS, PHASE2_EPOCHS, TOTAL_EPOCHS,
    PHASE1_LR, PHASE2_LR_BACKBONE, PHASE2_LR_HEAD,
    WARMUP_EPOCHS, WEIGHT_DECAY, LABEL_SMOOTHING,
    EARLY_STOP_PATIENCE, USE_AMP, AMP_DTYPE,
    SCHEDULER_T0, SCHEDULER_TMULT, SEED, MODEL_DIR,
    NUM_CLASSES, CLASS_NAMES, GRAD_CLIP_NORM,
)
from dataset import create_dataloaders
from models import HybridFusionModel, ImageOnlyModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()  # BN in backbone stays eval via override
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels, clinical in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with autocast("cuda", dtype=amp_dtype):
                outputs = model(images, clinical)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype):
    """Validate. Returns (avg_loss, accuracy, f1, auc, preds, labels, probs)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, clinical in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        if USE_AMP:
            with autocast("cuda", dtype=amp_dtype):
                outputs = model(images, clinical)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    try:
        auc = roc_auc_score(
            all_labels, np.array(all_probs), multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = 0.0

    return avg_loss, acc, f1, auc, all_preds, all_labels, all_probs


def train_image_only_model(train_loader, val_loader, criterion):
    """Train image-only model for Grad-CAM (same 2-phase approach)."""
    print("\n🔬 Training Image-Only model for Grad-CAM...")

    img_model = ImageOnlyModel(pretrained=True).to(DEVICE)

    # Phase 1: Freeze backbone
    img_model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, img_model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    print("  ── Phase 1: Head-only (5 epochs) ──")
    for epoch in range(1, 6):
        img_model.train()
        for images, labels, _ in tqdm(train_loader, desc=f"  ImgP1 E{epoch}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with autocast("cuda", dtype=AMP_DTYPE):
                    out = img_model(images)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(img_model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = img_model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

    # Phase 2: Unfreeze backbone with discriminative LR
    img_model.unfreeze_backbone()
    optimizer = optim.AdamW(
        img_model.get_param_groups(PHASE2_LR_BACKBONE, PHASE2_LR_HEAD),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler("cuda", enabled=USE_AMP)

    best_img_auc = 0.0
    print("  ── Phase 2: Full fine-tuning (15 epochs) ──")

    for epoch in range(1, 16):
        img_model.train()
        for images, labels, _ in tqdm(train_loader, desc=f"  ImgP2 E{epoch}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with autocast("cuda", dtype=AMP_DTYPE):
                    out = img_model(images)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(img_model.parameters(), max_norm=GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = img_model(images)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

        scheduler.step()

        # Validation
        img_model.eval()
        img_preds, img_labels, img_probs_list = [], [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels_np = labels.numpy()
                if USE_AMP:
                    with autocast("cuda", dtype=AMP_DTYPE):
                        out = img_model(images)
                else:
                    out = img_model(images)
                probs = torch.softmax(out.float(), dim=1).cpu().numpy()
                img_preds.extend(out.argmax(dim=1).cpu().numpy())
                img_labels.extend(labels_np)
                img_probs_list.extend(probs)

        try:
            img_auc = roc_auc_score(img_labels, np.array(img_probs_list), multi_class="ovr", average="macro")
        except ValueError:
            img_auc = 0.0
        img_acc = accuracy_score(img_labels, img_preds)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  ImgOnly Epoch {epoch}: Acc={img_acc:.4f}, AUC={img_auc:.4f}")

        if img_auc > best_img_auc:
            best_img_auc = img_auc
            torch.save(img_model.state_dict(), os.path.join(MODEL_DIR, "image_only_model_best.pth"))

    print(f"  ✅ Image-only model saved (best AUC: {best_img_auc:.4f})")
    return best_img_auc


def main():
    set_seed(SEED)

    print("🧠 Brain Stroke Prediction — Hybrid Model Training")
    print("=" * 60)
    print(f"   Device:          {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU:             {torch.cuda.get_device_name(0)}")
        print(f"   VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Mixed Precision: {USE_AMP} ({AMP_DTYPE})")
    print(f"   Phase 1:         {PHASE1_EPOCHS} epochs (frozen backbone, LR={PHASE1_LR})")
    print(f"   Phase 2:         {PHASE2_EPOCHS} epochs (unfrozen, bb LR={PHASE2_LR_BACKBONE}, head LR={PHASE2_LR_HEAD})")
    print(f"   Warmup:          {WARMUP_EPOCHS} epochs (linear)")
    print(f"   Backbone BN:     Always frozen (eval mode)")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names, class_weights = create_dataloaders()
    class_weights = class_weights.to(DEVICE)

    # ── Model ─────────────────────────────────────────────────────────
    print("\n🏗️  Building Hybrid Fusion Model...")
    model = HybridFusionModel(pretrained=True).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params:     {total_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    scaler = GradScaler("cuda", enabled=USE_AMP)

    # ── Training History ──────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": [],
        "phase": [], "lr_backbone": [], "lr_head": [],
    }
    global_epoch = 0
    best_model_path = os.path.join(MODEL_DIR, "hybrid_model_best.pth")

    # ═══════════════════════════════════════════════════════════════════
    #   PHASE 1: Freeze backbone — Train only heads
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔒 PHASE 1: Backbone FROZEN — Training heads only")
    print("=" * 60)

    model.freeze_backbone()

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p1, T_0=5, T_mult=1,
    )

    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable params: {trainable_p1:,} (heads only)")
    print(f"   Learning rate:    {PHASE1_LR}\n")

    for epoch in range(1, PHASE1_EPOCHS + 1):
        global_epoch += 1
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_p1, scaler, DEVICE, AMP_DTYPE
        )
        val_loss, val_acc, val_f1, val_auc, _, _, _ = validate(
            model, val_loader, criterion, DEVICE, AMP_DTYPE
        )
        scheduler_p1.step()

        lr = optimizer_p1.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["phase"].append(1)
        history["lr_backbone"].append(0.0)
        history["lr_head"].append(lr)

        print(
            f"[P1] Epoch {global_epoch:3d}/{TOTAL_EPOCHS} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f}  F1: {val_f1:.4f}  AUC: {val_auc:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1

    # ═══════════════════════════════════════════════════════════════════
    #   PHASE 2: Unfreeze backbone — Full fine-tuning
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("🔓 PHASE 2: Backbone UNFROZEN — Full fine-tuning (BN frozen)")
    print("=" * 60)

    # Reload best Phase 1 model before Phase 2
    print("   📂 Loading best Phase 1 checkpoint...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model.unfreeze_backbone()

    # Discriminative LR
    param_groups = model.get_param_groups(PHASE2_LR_BACKBONE, PHASE2_LR_HEAD)
    optimizer_p2 = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    scheduler_p2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=SCHEDULER_T0, T_mult=SCHEDULER_TMULT,
    )

    scaler = GradScaler("cuda", enabled=USE_AMP)
    patience_counter = 0  # Reset for Phase 2

    print(f"   Backbone LR:   {PHASE2_LR_BACKBONE}")
    print(f"   Head LR:       {PHASE2_LR_HEAD}")
    print(f"   Warmup:        {WARMUP_EPOCHS} epochs")
    print(f"   Backbone BN:   eval mode (stats preserved)\n")

    for epoch in range(1, PHASE2_EPOCHS + 1):
        global_epoch += 1
        t0 = time.time()

        # Linear warmup for first few epochs
        if epoch <= WARMUP_EPOCHS:
            warmup_factor = 0.1 + 0.9 * (epoch / WARMUP_EPOCHS)
            for pg in optimizer_p2.param_groups:
                base_lr = PHASE2_LR_BACKBONE if pg["name"] == "backbone" else PHASE2_LR_HEAD
                pg["lr"] = base_lr * warmup_factor

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_p2, scaler, DEVICE, AMP_DTYPE
        )
        val_loss, val_acc, val_f1, val_auc, _, _, _ = validate(
            model, val_loader, criterion, DEVICE, AMP_DTYPE
        )

        if epoch > WARMUP_EPOCHS:
            scheduler_p2.step()

        lr_bb = optimizer_p2.param_groups[0]["lr"]
        lr_hd = optimizer_p2.param_groups[1]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)
        history["phase"].append(2)
        history["lr_backbone"].append(lr_bb)
        history["lr_head"].append(lr_hd)

        print(
            f"[P2] Epoch {global_epoch:3d}/{TOTAL_EPOCHS} | "
            f"Train: {train_loss:.4f} / {train_acc:.4f} | "
            f"Val: {val_loss:.4f} / {val_acc:.4f}  F1: {val_f1:.4f}  AUC: {val_auc:.4f} | "
            f"bb: {lr_bb:.2e} hd: {lr_hd:.2e} | {elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  💾 Best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {global_epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    # ── Save Last Model & History ─────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "hybrid_model_last.pth"))
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    #   FINAL TEST EVALUATION
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 Final Test Set Evaluation")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    test_loss, test_acc, test_f1, test_auc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, DEVICE, AMP_DTYPE
    )

    print(f"\n  Test Accuracy:   {test_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"  Test AUC-ROC:    {test_auc:.4f}")

    final_class_names = class_names if class_names else CLASS_NAMES
    print(f"\n{classification_report(test_labels, test_preds, target_names=final_class_names)}")

    test_results = {
        "accuracy": float(test_acc),
        "f1_macro": float(test_f1),
        "auc_roc": float(test_auc),
        "best_val_auc": float(best_auc),
        "class_names": final_class_names,
        "total_epochs_trained": global_epoch,
    }
    with open(os.path.join(MODEL_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════
    #   TRAIN IMAGE-ONLY MODEL (for Grad-CAM)
    # ═══════════════════════════════════════════════════════════════════
    img_auc = train_image_only_model(train_loader, val_loader, criterion)

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"   Models saved to:        {MODEL_DIR}")
    print(f"   Best hybrid AUC:        {best_auc:.4f}")
    print(f"   Best image-only AUC:    {img_auc:.4f}")
    print(f"   Test Accuracy:          {test_acc:.4f}")
    print(f"   Test F1 (macro):        {test_f1:.4f}")
    print(f"   Epochs trained:         {global_epoch}")
    print(f"\n   Files saved:")
    print(f"     • hybrid_model_best.pth")
    print(f"     • hybrid_model_last.pth")
    print(f"     • image_only_model_best.pth")
    print(f"     • training_history.json")
    print(f"     • test_results.json")


if __name__ == "__main__":
    main()
