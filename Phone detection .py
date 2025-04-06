from ultralytics import YOLO
import cv2
import math
import time
import os

# Create alerts folder if it doesn't exist
if not os.path.exists("alerts"):
    os.makedirs("alerts")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Class names from YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# "cell phone" index
phone_class_id = classNames.index("cell phone")

last_alert_time = 0
alert_cooldown = 5  # seconds

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    phone_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # Skip if not a cell phone
            if cls != phone_class_id:
                continue

            phone_detected = True

            # Extract box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Draw box for cell phone only
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "cell phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

    # Show alert if phone detected
    current_time = time.time()
    if phone_detected:
        cv2.putText(img, "⚠️ WARNING: MOBILE PHONE DETECTED!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if current_time - last_alert_time > alert_cooldown:
            print("⚠️ ALERT: Mobile phone detected!")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"alerts/phone_{timestamp}.jpg", img)
            last_alert_time = current_time

    # Show camera feed
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# updated code
