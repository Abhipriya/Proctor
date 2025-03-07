import cv2
import mediapipe as mp
import numpy as np

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define landmark indices for eyes & iris
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Left eye contour
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Right eye contour
LEFT_IRIS = [468, 469, 470, 471, 472]  # Left iris
RIGHT_IRIS = [473, 474, 475, 476, 477]  # Right iris


# Function to determine gaze direction using iris position
def get_gaze_direction(landmarks, frame_shape):
    left_eye = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in LEFT_EYE])
    right_eye = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in RIGHT_EYE])

    left_iris = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in LEFT_IRIS])
    right_iris = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in RIGHT_IRIS])

    # Convert to pixel coordinates
    left_eye_px = left_eye * [frame_shape[1], frame_shape[0]]
    right_eye_px = right_eye * [frame_shape[1], frame_shape[0]]
    left_iris_px = left_iris * [frame_shape[1], frame_shape[0]]
    right_iris_px = right_iris * [frame_shape[1], frame_shape[0]]

    # Get iris center
    left_iris_center = np.mean(left_iris_px, axis=0)
    right_iris_center = np.mean(right_iris_px, axis=0)

    # Get eye center
    left_eye_center = np.mean(left_eye_px, axis=0)
    right_eye_center = np.mean(right_eye_px, axis=0)

    # Calculate iris movement relative to eye center
    left_movement = left_iris_center[0] - left_eye_center[0]
    right_movement = right_iris_center[0] - right_eye_center[0]

    # Define threshold for gaze direction
    if left_movement < -5 and right_movement < -5:
        return "Looking Left"
    elif left_movement > 5 and right_movement > 5:
        return "Looking Right"
    else:
        return "Looking Forward"


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            gaze_direction = get_gaze_direction(face_landmarks, frame.shape)

            # Draw text
            cv2.putText(frame, gaze_direction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Gaze Tracking (Iris-Based)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

