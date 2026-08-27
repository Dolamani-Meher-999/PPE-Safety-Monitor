import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import csv
import time

# Load trained model
model = YOLO(
    r"runs\detect\runs\detect\helmet_10epochs-2\weights\best.pt"
)

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# Folders
violation_folder = Path("violations")
violation_folder.mkdir(exist_ok=True)

# CSV log
log_file = Path("violations.csv")

if not log_file.exists():
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Date",
            "Time",
            "Violation",
            "Confidence",
            "Screenshot"
        ])

# Screenshot cooldown
last_violation_time = 0
COOLDOWN = 5

print("================================")
print(" PPE SAFETY MONITOR")
print("================================")
print("System started.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.40, verbose=False)

    no_helmet_detected = False
    highest_confidence = 0

    # Examine detections
    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        if class_name == "NoHelmet":

            no_helmet_detected = True

            if confidence > highest_confidence:
                highest_confidence = confidence

    # Draw YOLO boxes
    annotated_frame = results[0].plot()

    # ==================================
    # NO HELMET
    # ==================================

    if no_helmet_detected:

        cv2.putText(
            annotated_frame,
            "NO HELMET - SAFETY VIOLATION!",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3
        )

        current_time = time.time()

        # Save only once every 5 seconds
        if current_time - last_violation_time >= COOLDOWN:

            timestamp = datetime.now()

            filename = (
                violation_folder /
                f"violation_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            # Save screenshot
            cv2.imwrite(
                str(filename),
                annotated_frame
            )

            # Save CSV record
            with open(log_file, "a", newline="") as file:

                writer = csv.writer(file)

                writer.writerow([
                    timestamp.strftime("%Y-%m-%d"),
                    timestamp.strftime("%H:%M:%S"),
                    "NoHelmet",
                    f"{highest_confidence:.2f}",
                    str(filename)
                ])

            print(
                f"VIOLATION LOGGED | "
                f"Confidence: {highest_confidence:.2f}"
            )

            last_violation_time = current_time

    # ==================================
    # SAFE
    # ==================================

    else:

        cv2.putText(
            annotated_frame,
            "SAFETY OK",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )

    # Display webcam
    cv2.imshow(
        "PPE Safety Monitor",
        annotated_frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("System stopped.")