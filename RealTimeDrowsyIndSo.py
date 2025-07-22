import cv2
from tensorflow.keras.models import load_model  # type:ignore
import numpy as np
import dlib
import time
import winsound  

# Image dim
IMG_WIDTH, IMG_HEIGHT = 64, 64

model = load_model("Ver4/saved_model/CNNModel_128_5.h5")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Ver4/shape_predictor_68_face_landmarks.dat")  

def preprocess_eye_image(eye_img):
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = cv2.resize(eye_img, (IMG_WIDTH, IMG_HEIGHT))  
    eye_img = np.expand_dims(eye_img, axis=0) 
    eye_img = eye_img / 255.0  
    return eye_img

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  
    B = np.linalg.norm(eye[2] - eye[4])  
    C = np.linalg.norm(eye[0] - eye[3])  
    ear = (A + B) / (2.0 * C)  
    return ear


def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Check face
    if len(faces) == 0:
        return "No Face Detected", 0

    for face in faces:
        landmarks = predictor(gray, face)
        # Get the left and right eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_eye_img = frame[left_eye[0][1] - 18:left_eye[5][1] + 12, left_eye[0][0] - 12:left_eye[3][0] + 8]
        right_eye_img = frame[right_eye[0][1] - 18:right_eye[5][1] + 12, right_eye[0][0] - 12:right_eye[3][0] + 8]
       
        left_eye_input = preprocess_eye_image(left_eye_img)
        right_eye_input = preprocess_eye_image(right_eye_img)

        
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0

        
        input_data = np.array([[ear]])
        input_data = input_data.reshape(-1, 1)  

        # Normalize 
        prediction = model.predict(input_data)[0][0] 

        
        return "Drowsy" if prediction < 0.4 else "Alert", ear

# Initialize webcam
cap = cv2.VideoCapture(0)


drowsy_time = 0
is_drowsy = False

# Sound settings
sound_duration = 1000  
sound_frequency = 1000  

#  frame capture
while True:
    ret, frame = cap.read()
    if not ret:
        break
    label, prediction = detect_drowsiness(frame)
    color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    #cv2.putText(frame, f"Prediction: {prediction:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    
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

    # alarm soud
    if label == "Drowsy":
        if not is_drowsy:  
            is_drowsy = True
            drowsy_time = time.time()  # Rset  timer

        elif time.time() - drowsy_time > 1.5:  
            winsound.Beep(sound_frequency, sound_duration)  

    else:
        is_drowsy = False  

    cv2.imshow("Drowsiness Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
