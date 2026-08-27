import cv2
from ultralytics import YOLO

# Load our trained helmet detection model
model = YOLO(
    r"runs\detect\runs\detect\helmet_10epochs-2\weights\best.pt"
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")

while True:

    # Capture one frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Run YOLO on the frame
    results = model(frame, conf=0.40, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Display result
    cv2.imshow("Helmet Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam
cap.release()
cv2.destroyAllWindows()