import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model # type: ignore
import time
import winsound

# Image dimension 
IMG_WIDTH, IMG_HEIGHT = 128, 128 

# Load the trained CNN model for drowsiness detection
model = load_model("CNNModel_128.h5")

# Initialize the face and landmarks detector from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Ver5/shape_predictor_68_face_landmarks.dat")

#  preprocess 
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
        
        # left and right eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Extract and preprocess the eye images
        left_eye_img = frame[left_eye[1][1]:left_eye[5][1], left_eye[0][0]:left_eye[3][0]]
        right_eye_img = frame[right_eye[1][1]:right_eye[5][1], right_eye[0][0]:right_eye[3][0]]

        # Preprocess the eye images for the CNN model
        left_eye_input = preprocess_eye_image(left_eye_img)
        right_eye_input = preprocess_eye_image(right_eye_img)

        # Combine both eye images as input for the model
        eye_input = np.concatenate([left_eye_input, right_eye_input], axis=-1)  

        # Predict drowsiness using the CNN model
        prediction = model.predict(eye_input)[0][0]

        return "Drowsy" if prediction < 0.4 else "Alert", prediction

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

    #  drowsiness status and prediction
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
