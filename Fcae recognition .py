import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection and Drawing Utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load a sample image and preprocess for recognition
known_image = cv2.imread("Abhipriya.jpg")  # Replace with your image
known_image = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)

# Extract face using MediaPipe from the known image
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detector:
    results = face_detector.process(known_image)

if results.detections:
    for detection in results.detections:
        # Extract bounding box for the known face
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = known_image.shape
        top = int(bboxC.ymin * ih)
        left = int(bboxC.xmin * iw)
        bottom = int((bboxC.ymin + bboxC.height) * ih)
        right = int((bboxC.xmin + bboxC.width) * iw)

        # Crop and resize the face region
        known_face = known_image[top:bottom, left:right]
        known_face = cv2.resize(known_face, (100, 100))
        known_face_gray = cv2.cvtColor(known_face, cv2.COLOR_RGB2GRAY)

# Start the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform face detection using MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            top = int(bboxC.ymin * ih)
            left = int(bboxC.xmin * iw)
            bottom = int((bboxC.ymin + bboxC.height) * ih)
            right = int((bboxC.xmin + bboxC.width) * iw)

            # Extract face ROI
            face_roi = frame[top:bottom, left:right]
            if face_roi.size == 0:
                continue

            # Preprocess the face ROI
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            face_roi_gray = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)

            # Compare with the known face
            difference = cv2.absdiff(known_face_gray, face_roi_gray)
            score = np.mean(difference)

            # Threshold to identify the face
            name = "Unknown"
            if score < 50:  # Adjust the threshold as per requirement
                name = "Abhipriya"

            # Draw a bounding box and label using OpenCV
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam
video_capture.release()
cv2.destroyAllWindows()
