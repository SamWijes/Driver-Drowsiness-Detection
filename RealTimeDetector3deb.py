import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model
import time
import winsound

# Image dimensions
IMG_WIDTH, IMG_HEIGHT = 128, 128

# Load the trained CNN model for drowsiness detection
model = load_model("CNNModel_128_5.h5")

# Initialize the face and landmarks detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Ver5/shape_predictor_68_face_landmarks.dat")

# Preprocessing function for eye images
def preprocess_eye_image(eye_img):
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = cv2.resize(eye_img, (IMG_WIDTH, IMG_HEIGHT))  # Resize image
    eye_img = np.expand_dims(eye_img, axis=0) 
    eye_img = eye_img / 255.0  # Normalize to 0-1 range
    return eye_img

# Function to detect drowsiness
def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return "No Face Detected", 0

    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        left_eye_img = frame[left_eye[0][1] - 18:left_eye[5][1] + 12, left_eye[0][0] - 12:left_eye[3][0] + 8]
        right_eye_img = frame[right_eye[0][1] - 18:right_eye[5][1] + 12, right_eye[0][0] - 12:right_eye[3][0] + 8]

        # Debug: Visualize the extracted eye images
        cv2.imshow("Left Eye", left_eye_img)
        cv2.imshow("Right Eye", right_eye_img)

        left_eye_input = preprocess_eye_image(left_eye_img)
        right_eye_input = preprocess_eye_image(right_eye_img)

        # Debug: Print input data
        print("Left Eye Input Shape:", left_eye_input.shape, "Range:", left_eye_input.min(), left_eye_input.max())
        print("Right Eye Input Shape:", right_eye_input.shape, "Range:", right_eye_input.min(), right_eye_input.max())

        left_prediction = model.predict(left_eye_input)[0][0]
        right_prediction = model.predict(right_eye_input)[0][0]

        # Debug: Print predictions
        #print("Left Prediction:", left_prediction, "Right Prediction:", right_prediction)

        prediction = (left_prediction + right_prediction) / 2
        print("Prediction:",prediction)
        return "Drowsy" if prediction < 0.00743 else "Alert", prediction


# Initialize webcam
cap = cv2.VideoCapture(0)

drowsy_time = 0
is_drowsy = False

# Sound settings
sound_duration = 1000  # milliseconds
sound_frequency = 1000  # Hertz

# Frame capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get drowsiness prediction from the model
    label, prediction = detect_drowsiness(frame)

    # Set color based on prediction
    color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

    # Display drowsiness status and prediction
    cv2.putText(frame, f"Status: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Prediction: {prediction:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)

        # Left Eye
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        for point in left_eye:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

        # Right Eye
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        for point in right_eye:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

    # Check if the driver is drowsy and trigger sound alert
    if label == "Drowsy":
        if not is_drowsy:
            is_drowsy = True
            drowsy_time = time.time()  # Reset timer

        elif time.time() - drowsy_time > 1.5:  
            winsound.Beep(sound_frequency, sound_duration)

    else:
        is_drowsy = False

    # Show the processed frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
