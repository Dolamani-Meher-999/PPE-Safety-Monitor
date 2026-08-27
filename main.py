from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11n.pt")

# Detect objects
results = model("test.jpg")

# Save the result with bounding boxes
results[0].save(filename="result.jpg")

print("Result saved!")