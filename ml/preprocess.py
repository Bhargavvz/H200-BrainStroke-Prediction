"""
MRI Preprocessing Pipeline
Handles image transforms for training and inference:
  - Resize to 380×380
  - Intensity normalization (ImageNet stats)
  - Training augmentations (flip, rotate, color jitter, erasing)
"""
import torchvision.transforms as T
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms():
    """Training transforms with heavy augmentation."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """Validation / test transforms — only resize + normalize."""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms():
    """Inference transforms identical to val (for backend use)."""
    return get_val_transforms()
