import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

# State to track access restriction
test_access_restricted = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe processes RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    results = face_detection.process(rgb_frame)

    # Count the number of faces detected
    face_count = 0
    if results.detections:
        face_count = len(results.detections)

        # Draw face detections on the frame
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Restrict test access if more than one face is detected
    if face_count > 1:
        test_access_restricted = True
        cv2.putText(frame, "WARNING: Another person detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Test access restricted!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        test_access_restricted = False

    # Display test status
    if test_access_restricted:
        cv2.putText(frame, "Test Status: Restricted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Test Status: Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Online Proctoring System", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
