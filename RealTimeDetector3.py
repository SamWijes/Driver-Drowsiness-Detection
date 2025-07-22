import cv2
import numpy as np
import dlib
from tensorflow.keras.models import load_model # type: ignore
import time
import winsound

# Imag dim
IMG_WIDTH, IMG_HEIGHT = 128, 128

model = load_model("CNNModel_128_5.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Ver5/shape_predictor_68_face_landmarks.dat")


def preprocess_eye_image(eye_img):
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = cv2.resize(eye_img, (IMG_WIDTH, IMG_HEIGHT))  
    eye_img = np.expand_dims(eye_img, axis=0) 
    eye_img = eye_img / 255.0  
    return eye_img


def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return "No Face Detected", 0

    for face in faces:
        landmarks = predictor(gray, face)
        
        # Left and right eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_eye_img = frame[left_eye[0][1] - 18:left_eye[5][1] + 12, left_eye[0][0] - 12:left_eye[3][0] + 8]
        right_eye_img = frame[right_eye[0][1] - 18:right_eye[5][1] + 12, right_eye[0][0] - 12:right_eye[3][0] + 8]
       
        left_eye_input = preprocess_eye_image(left_eye_img)
        right_eye_input = preprocess_eye_image(right_eye_img)

        left_prediction = model.predict(left_eye_input)[0][0]
        right_prediction = model.predict(right_eye_input)[0][0]

        # Average the predictions 
        prediction = (left_prediction + right_prediction) / 2

        return "Drowsy" if prediction > 0.4 else "Alert", prediction

#  webcam
cap = cv2.VideoCapture(0)

drowsy_time = 0
is_drowsy = False

# Sound 
sound_duration = 1000  
sound_frequency = 1000  

# Frame capture
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #  prediction 
    label, prediction = detect_drowsiness(frame)

   
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

    # trigger sound alert
    if label == "Drowsy":
        if not is_drowsy:
            is_drowsy = True
            drowsy_time = time.time()  
        elif time.time() - drowsy_time > 1.5:  
            winsound.Beep(sound_frequency, sound_duration)

    else:
        is_drowsy = False

    
    cv2.imshow("Drowsiness Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
