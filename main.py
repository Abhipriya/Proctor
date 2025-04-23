import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import math
import os
from datetime import datetime
from collections import deque
import speech_recognition as sr
import langdetect
from ultralytics import YOLO
import sys
import requests
import signal
import logging

# ====================== Configuration ======================
BACKEND_URL = "http://127.0.0.1:8000/add-warning"
LOG_FILE = "proctoring_log.txt"
ALERTS_DIR = "alerts"
YOLO_WEIGHTS = "yolo-Weights/yolov8n.pt"
ALERT_COOLDOWN = 5  # seconds
ANGLE_BUFFER_SIZE = 5
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ====================== Logging Setup ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
    logging.FileHandler(LOG_FILE, encoding="utf-8"),
    logging.StreamHandler()
])

# ====================== Initialization ======================
# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Buffer for head angles
x_angles = deque(maxlen=ANGLE_BUFFER_SIZE)
y_angles = deque(maxlen=ANGLE_BUFFER_SIZE)

# Speech Recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

# YOLO Model
model = YOLO(YOLO_WEIGHTS)
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
phone_class_id = classNames.index("cell phone")

# Alert settings
os.makedirs(ALERTS_DIR, exist_ok=True)
last_alert_time = 0

# ====================== Helper Functions ======================
def send_warning_to_backend(warning_message):
    """Send a warning message to the backend."""
    try:
        response = requests.post(BACKEND_URL, json={"user_id": user_id, "message": warning_message})
        logging.info(f"[API] Warning sent: {response.status_code}")
    except Exception as e:
        logging.error(f"[API] Failed to send warning: {e}")

def log_event(message):
    """Log an event to the log file and console."""
    logging.info(message)

def handle_termination(signum, frame):
    """Handle termination signals for graceful shutdown."""
    logging.info("[INFO] Termination signal received. Cleaning up...")
    if 'cap' in globals():
        cap.release()
        cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_termination)
signal.signal(signal.SIGINT, handle_termination)

def detect_speech():
    """Continuously detect speech and check for non-English languages."""
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=3)
                text = recognizer.recognize_google(audio)
                detected_lang = langdetect.detect(text)
                log_event(f"Speech detected: {text} (Language: {detected_lang})")
                if detected_lang != "en":
                    log_event("⚠️ WARNING: Non-English speech detected!")
                    send_warning_to_backend("Non-English speech detected")
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                log_event("Error with speech recognition service.")
            except sr.WaitTimeoutError:
                pass

# ====================== Main Execution ======================
if __name__ == "__main__":
    # Get user_id from command line argument
    user_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"

    # Start speech detection thread
    speech_thread = threading.Thread(target=detect_speech, daemon=True)
    speech_thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        sys.exit(1)

    logging.info("Proctoring system started. Press 'q' to exit.")

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

        start = time.time()
        img_h, img_w, _ = image.shape
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results_mesh = face_mesh.process(rgb_frame)
        results_detection = face_detection.process(rgb_frame)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Face Count Detection
        face_count = 0
        face_visible = False

        if results_detection.detections:
            face_count = len(results_detection.detections)
            face_visible = True
            for detection in results_detection.detections:
                mp_drawing.draw_detection(image, detection)

        if results_mesh.multi_face_landmarks:
            face_visible = True

        if face_count > 1:
            log_event("⚠️ ALERT: Multiple faces detected! Test access restricted.")
            send_warning_to_backend("Multiple faces detected!")

        if not face_visible:
            log_event("⚠️ ALERT: No face detected! Test access restricted.")
            send_warning_to_backend("No face detected!")
            cv2.putText(image, "⚠️ WARNING: No face detected!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Head Pose Estimation
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                face_3d, face_2d = [], []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
                focal_length = img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

                x_angles.append(x)
                y_angles.append(y)
                avg_x = np.mean(x_angles)
                avg_y = np.mean(y_angles)

                # Show head pose direction
                if avg_y < -10:
                    direction = "Looking Left"
                elif avg_y > 10:
                    direction = "Looking Right"
                elif avg_x < -10:
                    direction = "Looking Down"
                elif avg_x > 10:
                    direction = "Looking Up"
                else:
                    direction = "Forward"

                cv2.putText(image, f"{direction}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.putText(image, f'Head X: {int(avg_x)}, Y: {int(avg_y)}', (20, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                if abs(avg_x) >= 16 or abs(avg_y) >= 16:
                    log_event("⚠️ ALERT: Excessive head movement detected!")
                    send_warning_to_backend("Excessive head movement detected!")
                    cv2.putText(image, "WARNING: Excessive movement!", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Phone Detection (YOLO)
        results = model(image, stream=True)
        phone_detected = False

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls != phone_class_id:
                    continue
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, "cell phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        current_time = time.time()
        if phone_detected and current_time - last_alert_time > ALERT_COOLDOWN:
            log_event("⚠️ ALERT: Mobile phone detected!")
            send_warning_to_backend("Mobile phone detected!")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"{ALERTS_DIR}/phone_{timestamp}.jpg", image)
            last_alert_time = current_time

        # Display
        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Proctoring System", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_event("Proctoring session ended by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

