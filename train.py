from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data=r"C:\Users\subha\OneDrive\Desktop\PPE- Safety- Manager\helmet_dataset\data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    project="runs/detect",
    name="helmet_10epochs"
)