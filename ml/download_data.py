"""
Brain Stroke CT Dataset Download & Organization
Downloads the Brain Stroke CT Dataset from Kaggle and organizes
it into train/val/test splits with stratified sampling.

Dataset: Brain Stroke CT Dataset (ozguraslank/brain-stroke-ct-dataset)
  - 6,653 CT brain slice images (annotated by 7 radiologists)
  - 3 classes: No Stroke (4,428), Ischemia (1,131), Bleeding (1,094)
  - Source: Turkish Ministry of Health (TEKNOFEST-2021)
  - Citation: Koç U, et al. Eurasian J Med., 2022;54(3):248-258
"""
import os
import shutil
import random
from pathlib import Path
from collections import Counter

from config import DATA_DIR, TRAIN_SPLIT, VAL_SPLIT, SEED, CLASS_NAMES

random.seed(SEED)


def download_dataset():
    """Download the dataset using kagglehub."""
    try:
        import kagglehub
        print("📥 Downloading Brain Stroke CT Dataset from Kaggle...")
        print("   Source: ozguraslank/brain-stroke-ct-dataset")
        print("   6,653 images · 3 classes · Annotated by 7 radiologists")
        path = kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"⚠️  kagglehub download failed: {e}")
        print("Trying alternative: looking for dataset in data/ directory...")
        # Check if dataset already exists
        possible_paths = [
            DATA_DIR,
            os.path.join(DATA_DIR, "brain-stroke-ct-image-dataset"),
        ]
        for p in possible_paths:
            if os.path.exists(p) and any(os.scandir(p)):
                print(f"✅ Found existing dataset at: {p}")
                return p
        raise RuntimeError(
            "Could not download dataset. Please download manually from:\n"
            "https://www.kaggle.com/datasets/ozguraslank/brain-stroke-ct-dataset\n"
            f"and extract it to: {DATA_DIR}"
        )


def find_image_root(download_path: str) -> str:
    """
    Find the root directory containing class folders.
    Kaggle datasets sometimes have nested directories.
    """
    download_path = str(download_path)

    # Look for directories that match our class names or common patterns
    for root, dirs, files in os.walk(download_path):
        # Check if this directory contains subdirectories that look like class folders
        lower_dirs = [d.lower() for d in dirs]
        has_normal = any("normal" in d or "no stroke" in d or "no_stroke" in d for d in lower_dirs)
        has_stroke = any("stroke" in d or "hemorrh" in d or "ischem" in d or "bleed" in d for d in lower_dirs)

        if has_normal and has_stroke:
            print(f"📂 Found image root: {root}")
            print(f"   Classes found: {dirs}")
            return root

    # If no class dirs found, maybe images are directly in the download path
    # Check for a single subfolder that contains class folders
    for item in Path(download_path).rglob("*"):
        if item.is_dir():
            subdirs = [d.name for d in item.iterdir() if d.is_dir()]
            lower_subdirs = [s.lower() for s in subdirs]
            has_n = any("normal" in s or "no stroke" in s or "no_stroke" in s for s in lower_subdirs)
            has_s = any("stroke" in s or "hemorrh" in s or "ischem" in s or "bleed" in s for s in lower_subdirs)
            if has_n and has_s:
                print(f"📂 Found image root: {item}")
                print(f"   Classes found: {subdirs}")
                return str(item)

    raise RuntimeError(f"Could not find class directories in: {download_path}")


def organize_dataset(image_root: str):
    """
    Organize images into train/val/test splits with stratified sampling.
    """
    output_dirs = {
        split: os.path.join(DATA_DIR, split)
        for split in ["train", "val", "test"]
    }

    # Clean existing splits
    for split_dir in output_dirs.values():
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # Collect all images per class
    class_images = {}
    class_dirs = [d for d in Path(image_root).iterdir() if d.is_dir()]

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
        if images:
            class_images[class_name] = images
            print(f"  📁 {class_name}: {len(images)} images")

    if not class_images:
        raise RuntimeError(f"No images found in {image_root}")

    # Stratified split
    total = 0
    for class_name, images in class_images.items():
        random.shuffle(images)
        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        # rest goes to test

        splits = {
            "train": images[:n_train],
            "val": images[n_train: n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            dest_dir = os.path.join(output_dirs[split_name], class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(str(img_path), dest_dir)
            total += len(split_images)

    # Print summary
    print("\n📊 Dataset Split Summary:")
    print("=" * 50)
    for split_name in ["train", "val", "test"]:
        split_dir = output_dirs[split_name]
        counts = {}
        for class_dir in Path(split_dir).iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.iterdir()))
                counts[class_dir.name] = count
        total_split = sum(counts.values())
        print(f"  {split_name:5s}: {total_split:5d} images — {counts}")
    print(f"  {'TOTAL':5s}: {total:5d} images")


def main():
    print("🧠 Brain Stroke MRI Dataset Setup")
    print("=" * 50)

    # Step 1: Download
    download_path = download_dataset()

    # Step 2: Find image root
    image_root = find_image_root(download_path)

    # Step 3: Organize into splits
    print("\n📂 Organizing into train/val/test splits...")
    organize_dataset(image_root)

    print("\n✅ Dataset ready for training!")
    print(f"   Location: {DATA_DIR}")


if __name__ == "__main__":
    main()
