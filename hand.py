import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Hand and Face modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Function to show image using matplotlib (in case cv2.imshow fails)
def show_with_matplotlib(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(rgb_frame)
    plt.axis('off')  # Hide axes
    plt.show()

# Using MediaPipe's Hand and Face modules
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands, \
     mp_face.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Convert the frame from BGR to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hands and face detection
        hand_results = hands.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)

        # Draw hand landmarks on the frame
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Draw face detection box on the frame
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Try to display the frame using cv2.imshow
        try:
            cv2.imshow("Hand and Face Recognition", frame)
        except cv2.error:
            # If cv2.imshow() fails, use matplotlib
            show_with_matplotlib(frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
