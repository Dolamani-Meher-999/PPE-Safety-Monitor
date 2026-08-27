from ultralytics import YOLO

model = YOLO(
    r"runs\detect\runs\detect\helmet_10epochs-2\weights\best.pt"
)

results = model.predict(
    source="helmet_test.jpg",
    conf=0.40,
    save=True
)

for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"Detected: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

print("Helmet test completed.")