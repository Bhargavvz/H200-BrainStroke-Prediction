"""
Evaluation & Visualization for Brain Stroke Prediction
Generates all required metrics & plots:
  - Confusion matrix heatmap
  - ROC-AUC curves (one-vs-rest, 3 classes)
  - Precision-Recall curves
  - Training vs Validation Loss & Accuracy curves
  - Per-class F1-score bar chart
  - Classification report
  - Sample Grad-CAM visualizations
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from torch.cuda.amp import autocast

from config import (
    DEVICE, MODEL_DIR, PLOT_DIR, CLASS_NAMES,
    NUM_CLASSES, USE_AMP, AMP_DTYPE,
)
from dataset import create_dataloaders
from models import HybridFusionModel, ImageOnlyModel
from grad_cam import GradCAM, overlay_heatmap
from preprocess import get_val_transforms
from PIL import Image

# ── Styling ───────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGSIZE = (10, 8)
DPI = 150


def load_best_model():
    """Load the best hybrid model."""
    model = HybridFusionModel(pretrained=False).to(DEVICE)
    state_dict = torch.load(
        os.path.join(MODEL_DIR, "hybrid_model_best.pth"),
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_test_predictions(model, test_loader):
    """Run model on test set, return labels, predictions, probabilities."""
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, clinical in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            clinical = clinical.to(DEVICE, non_blocking=True)

            if USE_AMP:
                with autocast(device_type="cuda", dtype=AMP_DTYPE):
                    outputs = model(images, clinical)
            else:
                outputs = model(images, clinical)

            probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(labels, preds, class_names):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_roc_curves(labels, probs, class_names):
    """Plot ROC-AUC curves for each class (one-vs-rest)."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC-AUC Curves (One-vs-Rest)", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "roc_auc_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_precision_recall_curves(labels, probs, class_names):
    """Plot Precision-Recall curves for each class."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i, (cls_name, color) in enumerate(zip(class_names, colors)):
        binary_labels = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, probs[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f"{cls_name} (AP = {pr_auc:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves", fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "precision_recall_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_training_curves():
    """Plot training vs validation loss & accuracy from saved history."""
    history_path = os.path.join(MODEL_DIR, "training_history.json")
    if not os.path.exists(history_path):
        print("  ⚠️  No training history found, skipping training curves.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=DPI)

    axes[0].plot(epochs, history["train_loss"], "b-", lw=2, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", lw=2, label="Val Loss")
    axes[0].set_title("Training vs Validation Loss", fontsize=15, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=13)
    axes[0].set_ylabel("Loss", fontsize=13)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], "b-", lw=2, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-", lw=2, label="Val Acc")
    axes[1].set_title("Training vs Validation Accuracy", fontsize=15, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=13)
    axes[1].set_ylabel("Accuracy", fontsize=13)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def plot_f1_bars(labels, preds, class_names):
    """Per-class F1-score bar chart."""
    f1s = f1_score(labels, preds, average=None)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(class_names, f1s, color=colors[:len(class_names)], edgecolor="white", width=0.6)

    for bar, f1_val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f1_val:.3f}", ha="center", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Scores", fontsize=16, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "f1_scores.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def generate_gradcam_samples(test_loader, class_names, num_samples=6):
    """Generate sample Grad-CAM visualizations from test set."""
    img_model_path = os.path.join(MODEL_DIR, "image_only_model_best.pth")
    if not os.path.exists(img_model_path):
        print("  ⚠️  Image-only model not found, skipping Grad-CAM samples.")
        return

    img_model = ImageOnlyModel(pretrained=False).to(DEVICE)
    img_model.load_state_dict(
        torch.load(img_model_path, map_location=DEVICE, weights_only=True)
    )
    img_model.eval()

    cam = GradCAM(img_model)

    # Collect sample images
    transform = get_val_transforms()
    samples_collected = 0
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8), dpi=DPI)

    for images, labels, _ in test_loader:
        for i in range(min(images.size(0), num_samples - samples_collected)):
            img_tensor = images[i].unsqueeze(0)
            label = labels[i].item()

            # Generate Grad-CAM
            heatmap, pred_class, confidence = cam.generate(img_tensor)

            # Denormalize for display
            img_display = images[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_display = (img_display * std + mean).clip(0, 1)
            img_display = (img_display * 255).astype(np.uint8)

            # Overlay
            overlay = overlay_heatmap(img_display, heatmap, alpha=0.5)

            # Plot original
            axes[0, samples_collected].imshow(img_display)
            axes[0, samples_collected].set_title(
                f"True: {class_names[label]}", fontsize=10, fontweight="bold"
            )
            axes[0, samples_collected].axis("off")

            # Plot Grad-CAM overlay
            axes[1, samples_collected].imshow(overlay)
            axes[1, samples_collected].set_title(
                f"Pred: {class_names[pred_class]}\n"
                f"Conf: {confidence:.2%}", fontsize=10
            )
            axes[1, samples_collected].axis("off")

            samples_collected += 1
            if samples_collected >= num_samples:
                break
        if samples_collected >= num_samples:
            break

    axes[0, 0].set_ylabel("Original", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=14, fontweight="bold")

    plt.suptitle("Grad-CAM Explainability — Sample MRI Predictions",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "gradcam_samples.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


def main():
    print("📊 Brain Stroke Prediction — Evaluation & Visualization")
    print("=" * 60)

    # Load data
    _, _, test_loader, class_names, _ = create_dataloaders()
    if not class_names:
        class_names = CLASS_NAMES

    # Load model
    model = load_best_model()
    print(f"  ✅ Loaded best hybrid model from {MODEL_DIR}")

    # Get predictions
    print("\n🔍 Running inference on test set...")
    labels, preds, probs = get_test_predictions(model, test_loader)

    # ── Metrics ───────────────────────────────────────────────────────
    print("\n📈 Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    try:
        auc_score = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_score = 0.0

    print(f"  Overall Accuracy:   {acc:.4f}")
    print(f"  Macro F1:           {f1_macro:.4f}")
    print(f"  Macro AUC-ROC:      {auc_score:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────
    print(f"\n🎨 Generating visualizations in {PLOT_DIR}...")

    plot_confusion_matrix(labels, preds, class_names)
    plot_roc_curves(labels, probs, class_names)
    plot_precision_recall_curves(labels, probs, class_names)
    plot_training_curves()
    plot_f1_bars(labels, preds, class_names)
    generate_gradcam_samples(test_loader, class_names)

    print(f"\n✅ All visualizations saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
