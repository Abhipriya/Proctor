import cv2#face detection
import mediapipe as mp
import numpy as np
from datetime import datetime

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Face detection parameters
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Log file for tracking events
log_file = "proctoring_log.txt"

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to log events
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as log:
        log.write(f"[{timestamp}] {message}\n")
    print(f"{timestamp} - {message}")

# Initialize test state
multi_person_detected = False
face_not_detected = False

print("Proctoring system started. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    # Count detected faces
    face_count = 0
    if results.detections:
        face_count = len(results.detections)

        # Draw detections on the frame
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Check for multiple faces
    if face_count > 1:
        if not multi_person_detected:
            log_event("Multiple faces detected! Test access restricted.")
            multi_person_detected = True
        cv2.putText(frame, "WARNING: Multiple faces detected!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        multi_person_detected = False

    # Check if no face is detected
    if face_count == 0:
        if not face_not_detected:
            log_event("No face detected! Test access restricted.")
            face_not_detected = True
        cv2.putText(frame, "WARNING: No face detected!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        face_not_detected = False

    # Display test status
    if multi_person_detected or face_not_detected:
        cv2.putText(frame, "Test Status: Restricted", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Test Status: Active", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Proctoring System", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        log_event("Proctoring session ended by user.")
        break

cap.release()
cv2.destroyAllWindows()
