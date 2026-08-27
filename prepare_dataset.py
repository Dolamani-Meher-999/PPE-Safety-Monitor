from pathlib import Path
import shutil
import random

# Original dataset
source = Path(
    r"C:\Users\subha\OneDrive\Desktop\EdgeVision Dataset\EdgeVision Dataset\EdgeVision-Dataset"
)

# New dataset for training
destination = Path(
    r"C:\Users\subha\OneDrive\Desktop\PPE- Safety- Manager\helmet_dataset"
)

images_source = source / "images"
labels_source = source / "labels" / "yolo"

# Create folders
for split in ["train", "val", "test"]:
    (destination / "images" / split).mkdir(parents=True, exist_ok=True)
    (destination / "labels" / split).mkdir(parents=True, exist_ok=True)

# Get all images
images = list(images_source.glob("*.jpg"))

# Shuffle in a reproducible way
random.seed(42)
random.shuffle(images)

# Split 80 / 10 / 10
total = len(images)

train_end = int(total * 0.80)
val_end = train_end + int(total * 0.10)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

splits = {
    "train": train_images,
    "val": val_images,
    "test": test_images
}

# Copy images and matching labels
for split, split_images in splits.items():

    print(f"\nPreparing {split}: {len(split_images)} images")

    for image in split_images:

        label = labels_source / f"{image.stem}.txt"

        if not label.exists():
            print(f"WARNING: Missing label for {image.name}")
            continue

        shutil.copy2(
            image,
            destination / "images" / split / image.name
        )

        shutil.copy2(
            label,
            destination / "labels" / split / label.name
        )

print("\nDataset preparation completed!")

print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))

print("\nNew dataset:")
print(destination)