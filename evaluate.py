from ultralytics import YOLO

# Load the NEW 10-epoch model
model = YOLO(
    r"C:\Users\subha\OneDrive\Desktop\PPE- Safety- Manager\PPE- Safety\runs\detect\helmet_10epochs\weights\best.pt"
)

# Evaluate on the TEST dataset
results = model.val(
    data=r"C:\Users\subha\OneDrive\Desktop\PPE- Safety- Manager\helmet_dataset\data.yaml",
    split="test",
    imgsz=640,
    batch=4,
    device="cpu"
)

print("\n===== 10 EPOCH TEST RESULTS =====")
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
print("Precision:", results.box.mp)
print("Recall:", results.box.mr)