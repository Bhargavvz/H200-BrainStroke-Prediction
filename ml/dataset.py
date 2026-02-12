"""
PyTorch Dataset & DataLoader for Brain Stroke MRI images.
Supports loading images with optional dummy clinical features.
Handles class-weighted sampling for imbalanced data.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from collections import Counter
from pathlib import Path

from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    NUM_CLINICAL_FEATURES, SEED,
)
from preprocess import get_train_transforms, get_val_transforms


class BrainStrokeMRIDataset(Dataset):
    """
    Brain Stroke MRI Dataset.
    Loads images from directory structure: split/class_name/image.jpg
    Returns (image_tensor, label, clinical_features).
    Clinical features are zeros during training (no metadata available).
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []      # list of (image_path, label_index)
        self.class_names = []   # sorted class names

        # Discover classes from directory names
        class_dirs = sorted([
            d for d in Path(root_dir).iterdir() if d.is_dir()
        ])
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Collect all image paths
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    self.samples.append((str(img_file), label))

        self.labels = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image and convert to RGB (handles grayscale MRI)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Dummy clinical features (zeros) — will be real during inference
        clinical = torch.zeros(NUM_CLINICAL_FEATURES, dtype=torch.float32)

        return image, label, clinical

    def get_class_weights(self):
        """Compute inverse-frequency class weights for loss function."""
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = []
        for i in range(len(self.class_names)):
            w = total / (len(self.class_names) * counts.get(i, 1))
            weights.append(w)
        return torch.FloatTensor(weights)


def get_weighted_sampler(dataset: BrainStrokeMRIDataset):
    """Create a WeightedRandomSampler to handle class imbalance."""
    counts = Counter(dataset.labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def create_dataloaders():
    """
    Create train/val/test DataLoaders with appropriate transforms.
    Returns: (train_loader, val_loader, test_loader, class_names, class_weights)
    """
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # Verify directories exist
    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"{name} directory not found at {d}. "
                "Run download_data.py first!"
            )

    train_dataset = BrainStrokeMRIDataset(train_dir, transform=get_train_transforms())
    val_dataset = BrainStrokeMRIDataset(val_dir, transform=get_val_transforms())
    test_dataset = BrainStrokeMRIDataset(test_dir, transform=get_val_transforms())

    print(f"📊 Dataset sizes — Train: {len(train_dataset)}, "
          f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"   Classes: {train_dataset.class_names}")

    # Weighted sampler for training
    train_sampler = get_weighted_sampler(train_dataset)
    class_weights = train_dataset.get_class_weights()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader, train_dataset.class_names, class_weights
