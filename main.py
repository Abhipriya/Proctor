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

# ====================== Initialization ======================
# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
log_file = "proctoring_log.txt"

# Buffer for head angles
angle_buffer_size = 5
x_angles = deque(maxlen=angle_buffer_size)
y_angles = deque(maxlen=angle_buffer_size)

# Speech Recognition
recognizer = sr.Recognizer()
mic = sr.Microphone()

# YOLO Model
model = YOLO("yolo-Weights/yolov8n.pt")
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
if not os.path.exists("alerts"):
    os.makedirs("alerts")
last_alert_time = 0
alert_cooldown = 5  # seconds

# ====================== Functions ======================
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as log:  # <-- UTF-8 encoding added here
        log.write(f"[{timestamp}] {message}\n")
    print(f"{timestamp} - {message}")


def detect_speech():
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for speech...")
            try:
                audio = recognizer.listen(source, timeout=3)
                text = recognizer.recognize_google(audio)
                detected_lang = langdetect.detect(text)
                log_event(f"Speech detected: {text} (Language: {detected_lang})")
                if detected_lang != "en":
                    log_event("WARNING: Non-English speech detected!")
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                log_event("Error with speech recognition service.")
            except sr.WaitTimeoutError:
                pass

# Start speech detection thread
speech_thread = threading.Thread(target=detect_speech, daemon=True)
speech_thread.start()

# ====================== Main Execution ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

multi_person_detected = False
face_not_detected = False

print("Proctoring system started. Press 'q' to exit.")

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    start = time.time()
    img_h, img_w, _ = image.shape
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results_mesh = face_mesh.process(rgb_frame)
    results_detection = face_detection.process(rgb_frame)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ================== Face Count Detection ==================
    face_count = 0
    if results_detection.detections:
        face_count = len(results_detection.detections)
        for detection in results_detection.detections:
            mp_drawing.draw_detection(image, detection)

    if face_count > 1 and not multi_person_detected:
        log_event("Multiple faces detected! Test access restricted.")
        multi_person_detected = True
    elif face_count <= 1:
        multi_person_detected = False

    if face_count == 0 and not face_not_detected:
        log_event("No face detected! Test access restricted.")
        face_not_detected = True
    elif face_count > 0:
        face_not_detected = False

    # ================== Head Pose Estimation ==================
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

            if avg_y < -10:
                text = "Looking Left"
            elif avg_y > 10:
                text = "Looking Right"
            elif avg_x < -10:
                text = "Looking Down"
            elif avg_x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

            if abs(avg_x) >= 17 or abs(avg_y) >= 15:
                log_event("Excessive head movement detected!")
                cv2.putText(image, "WARNING: Excessive movement detected!", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ================== Phone Detection (YOLO) ==================
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
    if phone_detected:
        cv2.putText(image, "⚠️ WARNING: MOBILE PHONE DETECTED!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if current_time - last_alert_time > alert_cooldown:
            log_event("⚠️ ALERT: Mobile phone detected!")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"alerts/phone_{timestamp}.jpg", image)
            last_alert_time = current_time

    # ================== Display ==================
    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("Proctoring System", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        log_event("Proctoring session ended by user.")
        break

cap.release()
cv2.destroyAllWindows()
