# Real-Time Drowsiness Detection (CNN + Dlib + Webcam)
This is my final year project on Bsc Software Engineering (2024)
A **real-time drowsiness detection system** built using a **CNN-based approach** and tested live on a **webcam feed**.  
This project detects eye state / drowsiness by isolating the eye region using **Dlib facial landmarks** (more reliable in low-light compared to Haar cascades) and running inference using a trained **7-layer CNN model** trained on the **MRL Eye Dataset**.

---

## Features
- Real-time webcam detection (OpenCV)
- Eye region extraction using **Dlib 68 facial landmark predictor**
- Better accuracy in low-light by cropping eye ROI from landmark points (instead of Haar cascade)
- CNN model training pipeline included
- Alarm (beep) when drowsiness is detected continuously

---

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Dlib (face detector + 68 landmarks)
- NumPy

---

## 📂 Dataset
- **MRL Eye Dataset** (open/closed eye images)

> Dataset is not included in this repository. Please download it separately available in kaggle or https://mrl.cs.vsb.cz//index.html

---


## ⚙️ Installation
1. Clone repository
````
  git clone https://github.com/SamWijes/Driver-Drowsiness-Detection.git
````
3. Install dependencies
````
pip install opencv-python numpy matplotlib tensorflow keras dlib
````

Note (Windows): Installing dlib can be tricky. If you get errors, install cmake and Visual Studio Build Tools, then retry.

## 🏋️ Training the CNN Model

Put dataset images in ./images/ with subfolders like:
-images/open/
-images/closed/

Run training:
python train_model.py

Training Notes
- Uses ImageDataGenerator(validation_split=0.2)
- Input size: 128×128 grayscale
> Use 64*64 if you have a lower end processor training time can take upto 45mins
- Output: binary classification with sigmoid
--

Saves:
model checkpoint: cp.keras
final model: CNNModel_128_3.h5
training history JSON: training_history_3.json
>Save important data into a json for analyzys and presentation

## 🎥 Real-Time Drowsiness Detection (Webcam)
- Requirements
- Trained model file (example):
- /saved_model/CNNModel_128.h5
- Dlib landmarks file:
- Ver4/shape_predictor_68_face_landmarks.dat

## Run detection
```
python detect_drowsiness.py
```
## Controls
Press q to quit the webcam window.

## Alarm Logic
If the system predicts Drowsy continuously for ~1.5 seconds, it triggers a beep alarm.

## Results / Observations
Dlib landmark-based ROI extraction provides more stable eye detection compared to Haar cascades, especially under poor lighting conditions.
Works best when the face is clearly visible and within a reasonable distance from the camera.

## 🚀 Future Improvements
Eventhough the setup proved to function with accuracy close to 90% it had difficulty in detecting underlowlight conditions there fore the setup needs a IR/night vision cam to realize its full potential.
Also rather than using a beep a interactive driver assist can be introduced to keep driver engaged via voice chat if continuous drowsyness detected

## 🙌 Credits
MRL Eye Dataset (for training images)
Dlib (face landmarks)
TensorFlow/Keras + OpenCV community

## 📜 License

MIT License



