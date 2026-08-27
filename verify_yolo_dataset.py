from pathlib import Path

dataset = Path(
    r"C:\Users\subha\OneDrive\Desktop\PPE- Safety- Manager\helmet_dataset"
)

print("Dataset exists:", dataset.exists())
print("data.yaml exists:", (dataset / "data.yaml").exists())

for split in ["train", "val", "test"]:
    image_folder = dataset / "images" / split
    label_folder = dataset / "labels" / split

    images = list(image_folder.glob("*.jpg"))
    labels = list(label_folder.glob("*.txt"))

    image_names = {x.stem for x in images}
    label_names = {x.stem for x in labels}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(missing_labels))
    print("Missing images:", len(missing_images))

print("\nClasses:")
print("0 = BikeWithRider")
print("1 = NoHelmet")
print("2 = Helmet")