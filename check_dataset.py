# from pathlib import Path

# dataset = Path(
#     r"C:\Users\subha\OneDrive\Desktop\EdgeVision Dataset\EdgeVision Dataset\EdgeVision-Dataset"
# )

# images_folder = dataset / "images"
# labels_folder = dataset / "labels" / "yolo"

# print("Dataset exists:", dataset.exists())
# print("Images folder exists:", images_folder.exists())
# print("Labels folder exists:", labels_folder.exists())

# images = list(images_folder.glob("*.jpg"))
# labels = list(labels_folder.glob("*.txt"))

# print("Images:", len(images))
# print("Labels:", len(labels))

from pathlib import Path

dataset = Path(
    r"C:\Users\subha\OneDrive\Desktop\EdgeVision Dataset\EdgeVision Dataset\EdgeVision-Dataset"
)

images_folder = dataset / "images"
labels_folder = dataset / "labels" / "yolo"

images = {file.stem for file in images_folder.glob("*.jpg")}
labels = {file.stem for file in labels_folder.glob("*.txt")}

missing_image = labels - images
missing_label = images - labels

print("Images:", len(images))
print("Labels:", len(labels))

print("\nLabels without images:")

for name in sorted(missing_image):
    print(name)

print("\nImages without labels:")

for name in sorted(missing_label):
    print(name)