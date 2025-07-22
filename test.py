import cv2
import dlib
import numpy as np

# Load the image
image_path = 'Ver4\man.jpg'  # Provide the path to your sample image
image = cv2.imread(image_path)

# Load the pre-trained dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Ver5/shape_predictor_68_face_landmarks.dat")  # Path to the shape predictor file

# Load the pre-trained dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Ver5/shape_predictor_68_face_landmarks.dat") 


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = detector(gray)

if len(faces) == 0:
    print("No faces detected")
else:
    
    for face in faces:
        
        landmarks = predictor(gray, face)
        
        # 36-41 for left42-47  right )
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        
        
        #left_eye_img = image[left_eye[0][1] - 10:left_eye[5][1] + 10, left_eye[0][0] - 10:left_eye[3][0] + 10]
        left_eye_img = image[left_eye[0][1] - 18:left_eye[5][1] + 12, left_eye[0][0] - 12:left_eye[3][0] + 8]
        
        right_eye_img = image[right_eye[0][1] - 18:right_eye[5][1] + 12, right_eye[0][0] - 12:right_eye[3][0] + 8]
        #right_eye_img = image[right_eye[0][1] - 5:right_eye[5][1] + 10, right_eye[0][0] - 3:right_eye[3][0] + 3]

        
        left_eye_resized = cv2.resize(left_eye_img, (128, 128))
        right_eye_resized = cv2.resize(right_eye_img, (128, 128))

        
        cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL)
        cv2.imshow("Left Eye", left_eye_resized)

        cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL)
        cv2.imshow("Right Eye", right_eye_resized)

        
        cv2.imwrite("left_eye_resized.jpg", left_eye_resized)
        cv2.imwrite("right_eye_resized.jpg", right_eye_resized)

        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
