from tensorflow.keras.models import load_model
import cv2

import time

import numpy as np
from tensorflow.keras.preprocessing import image


# Load the trained model
model = load_model('./CNN_7_Layers.h5')


# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()



def preprocess_frame(frame):
    # Resize the frame to the size expected by the model (224x224)
    frame_resized = cv2.resize(frame, (224, 224))
    
    # Convert the frame to a format compatible with the model (float32 and normalized)
    frame_resized = np.expand_dims(frame_resized, axis=0)  # Add batch dimension
    frame_resized = frame_resized / 255.0  # Normalize pixel values (if not already normalized)

    return frame_resized


def predict_eye_state(frame):
    # Preprocess the frame for prediction
    preprocessed_frame = preprocess_frame(frame)

    # Get the model's prediction
    prediction = model.predict(preprocessed_frame)
    
    # If the prediction is close to 1, the eyes are open; if close to 0, eyes are closed
    return prediction[0][0]  # Return the prediction score for 'open' vs 'closed'



# Define thresholds
EYE_CLOSED_THRESHOLD = 0.2  # Predicts eyes closed if the output is below this threshold
DROWSINESS_ALERT_TIME = 3   # Time (in seconds) for which eyes must stay closed to trigger an alert

eye_closed_start_time = None  # Keep track of when the eyes were first detected closed
is_drowsy = False

# Process the video stream
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Check if frame is correctly captured
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Predict the eye state
    eye_state = predict_eye_state(frame)

    if eye_state < EYE_CLOSED_THRESHOLD:
        # If eyes are closed, check if they have been closed long enough to trigger an alert
        if eye_closed_start_time is None:
            eye_closed_start_time = time.time()  # Set the start time when eyes are first closed
        elif time.time() - eye_closed_start_time >= DROWSINESS_ALERT_TIME:
            is_drowsy = True  # Drowsy if eyes are closed for long enough
    else:
        # Reset the timer if eyes are open
        eye_closed_start_time = None
        is_drowsy = False
    
    # Display the drowsiness alert on the frame
    if is_drowsy:
        cv2.putText(frame, "Drowsiness Alert!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
