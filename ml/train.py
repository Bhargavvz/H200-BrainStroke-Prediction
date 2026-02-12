"""
Training Script for Brain Stroke Prediction Hybrid Model
  - Mixed precision (BF16/FP16) for GPU acceleration
  - Cosine annealing LR scheduler with warm restarts
  - Early stopping
  - Class-weighted cross-entropy loss with label smoothing
  - Saves best model by validation AUC-ROC
"""
import os
import sys
import time
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report,
)
from tqdm import tqdm

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LABEL_SMOOTHING, EARLY_STOP_PATIENCE, USE_AMP, AMP_DTYPE,
    SCHEDULER_T0, SCHEDULER_TMULT, SEED, MODEL_DIR,
    NUM_CLASSES, CLASS_NAMES,
)
from dataset import create_dataloaders
from models import HybridFusionModel, ImageOnlyModel


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels, clinical in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            with autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images, clinical)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    """Validate the model. Returns (avg_loss, accuracy, f1, auc, all_preds, all_labels, all_probs)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels, clinical in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)

        if USE_AMP:
            with autocast(device_type="cuda", dtype=amp_dtype):
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

    # AUC-ROC (one-vs-rest)
    try:
        all_probs_np = np.array(all_probs)
        auc = roc_auc_score(
            all_labels, all_probs_np, multi_class="ovr", average="macro"
        )
    except ValueError:
        auc = 0.0

    return avg_loss, acc, f1, auc, all_preds, all_labels, all_probs


def main():
    set_seed(SEED)
    print("🧠 Brain Stroke Prediction — Hybrid Model Training")
    print("=" * 60)
    print(f"   Device:          {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU:             {torch.cuda.get_device_name(0)}")
        print(f"   VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   Mixed Precision: {USE_AMP} ({AMP_DTYPE})")
    print(f"   Epochs:          {EPOCHS}")
    print(f"   Learning Rate:   {LEARNING_RATE}")
    print()

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names, class_weights = create_dataloaders()
    class_weights = class_weights.to(DEVICE)

    # ── Model ─────────────────────────────────────────────────────────
    print("\n🏗️  Building Hybrid Fusion Model...")
    model = HybridFusionModel(pretrained=True).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # Try torch.compile for PyTorch 2.x speedup
    try:
        model = torch.compile(model)
        print("   ✅ torch.compile() enabled")
    except Exception:
        print("   ⚠️  torch.compile() not available, proceeding without")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_TMULT,
    )

    scaler = GradScaler(enabled=USE_AMP)

    # ── Training Loop ─────────────────────────────────────────────────
    best_auc = 0.0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": [],
    }

    print(f"\n🚀 Starting training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, AMP_DTYPE
        )

        # Validate
        val_loss, val_acc, val_f1, val_auc, _, _, _ = validate(
            model, val_loader, criterion, DEVICE, AMP_DTYPE
        )

        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
            f"F1: {val_f1:.4f}  AUC: {val_auc:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "hybrid_model_best.pth"))
            print(f"  💾 Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    # ── Save Last Model & History ─────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "hybrid_model_last.pth"))

    # Save training history
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Final Test Evaluation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 Final Test Set Evaluation")
    print("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "hybrid_model_best.pth"), weights_only=True))
    test_loss, test_acc, test_f1, test_auc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, DEVICE, AMP_DTYPE
    )

    print(f"\n  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"  Test AUC-ROC:   {test_auc:.4f}")

    # Determine final class_names from actual data
    final_class_names = class_names if class_names else CLASS_NAMES

    print(f"\n{classification_report(test_labels, test_preds, target_names=final_class_names)}")

    # Save test results
    test_results = {
        "accuracy": float(test_acc),
        "f1_macro": float(test_f1),
        "auc_roc": float(test_auc),
        "class_names": final_class_names,
    }
    with open(os.path.join(MODEL_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # ── Train Image-Only Model (for Grad-CAM) ────────────────────────
    print("\n🔬 Training Image-Only model for Grad-CAM...")
    img_model = ImageOnlyModel(pretrained=True).to(DEVICE)

    try:
        img_model = torch.compile(img_model)
    except Exception:
        pass

    img_optimizer = optim.AdamW(img_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    img_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        img_optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_TMULT
    )
    img_scaler = GradScaler(enabled=USE_AMP)

    best_img_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        img_model.train()
        for images, labels, _ in tqdm(train_loader, desc=f"  ImgOnly E{epoch}", leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            img_optimizer.zero_grad(set_to_none=True)
            if USE_AMP:
                with autocast(device_type="cuda", dtype=AMP_DTYPE):
                    outputs = img_model(images)
                    loss = criterion(outputs, labels)
                img_scaler.scale(loss).backward()
                img_scaler.step(img_optimizer)
                img_scaler.update()
            else:
                outputs = img_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                img_optimizer.step()

        img_scheduler.step()

        # Quick validation
        img_model.eval()
        img_preds, img_labels, img_probs_list = [], [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels_np = labels.numpy()
                if USE_AMP:
                    with autocast(device_type="cuda", dtype=AMP_DTYPE):
                        outputs = img_model(images)
                else:
                    outputs = img_model(images)
                probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
                img_preds.extend(outputs.argmax(dim=1).cpu().numpy())
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

    print("\n✅ Training complete!")
    print(f"   Models saved to: {MODEL_DIR}")
    print(f"   Best hybrid AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
