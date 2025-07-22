import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# Set up the directory for your data
data_dir = 'C:/Users/samit/Documents/CODE/IndProjectCode/Ver4/images2'

def preprocessing(img):
    return img / 255.0

# Data Split/pre pocess
datagen = ImageDataGenerator(validation_split=0.2,preprocessing_function=preprocessing)

# Data generators for training and validation
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),  
    batch_size=64,           
    shuffle=True,
    class_mode='binary',
    subset='training',
    color_mode='grayscale'
)

print("Number of batches in training data:", len(train_generator))

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),  
    batch_size=64,
    class_mode='binary',
    subset='validation',
    color_mode='grayscale'
)

print("Number of batches in testing data:", len(test_generator))

#test_generator = test_generator.repeat()

# CNN model
CNNmodel = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    
    layers.GlobalAveragePooling2D(),  
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(128, activation='relu', kernel_regularizer='l1'),
    
    layers.Dense(1, activation='sigmoid')  
])


CNNmodel.summary()

# Compile 
CNNmodel.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

# Save the best model
filepath = 'C:/Users/samit/Documents/CODE/IndProjectCode/Ver5/cp.keras'
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max', 
                             verbose=1,
                             save_freq=5)

# Early stopping 
early_stopping = EarlyStopping(
                    monitor='val_accuracy',  
                    patience=5,              
                    min_delta=0.0,           
                    verbose=1,               
                    mode='max',
                    baseline=0.85              
            )            




# Fit the model 
history = CNNmodel.fit(
    train_generator,
    epochs=3,
     
    validation_data=test_generator,
     
    callbacks=checkpoint
)


CNNmodel.save('CNNModel_128_3.h5')

# Save the training history to a JSON file
history_file = 'training_history_3.json'
with open(history_file, 'w') as f:
    json.dump(history.history, f)

print(f"Training history saved to {history_file}")

'''
best = load_model(filepath)

test_accuracy = best.evaluate(test_generator)'''
