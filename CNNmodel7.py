
import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib.image import imread

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models, layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG19

from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.preprocessing import image


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# 
data_dir = './images'

datagen = ImageDataGenerator(validation_split=0.3, preprocessing_function=preprocess_input)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=64,
    shuffle=True,
    class_mode='binary',
    subset='training'
)

print("Number of batches in training data :", len(train_generator))

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    subset='validation'
) 


print("Number of batches in testing data :", len(test_generator))

CNNmodel_7 = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'),
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
    
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(1024, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Reduce the kernel size here to avoid output size issues
    layers.Conv2D(2048, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(128, activation='relu', kernel_regularizer='l1'),
    
    layers.Dense(1, activation='sigmoid')
])


CNNmodel_7.summary()

steps_per_epoch_training = len(train_generator)
steps_per_epoch_testing = len(test_generator)

CNNmodel_7.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

filepath = './cp.keras'

checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max', 
                             verbose=1)
from tensorflow.keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(
                    monitor='val_accuracy',  
                    patience=5,              # Stop after 5 epochs without improvement
                    min_delta=0.0,           # Any change in accuracy is considered an improvement
                    verbose=1,               
                    mode='max',              
                    baseline=0.85            # Early stopping only triggers if validation accuracy exceeds 99%
)


history = CNNmodel_7.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    callbacks=[early_stopping,checkpoint]
)

CNNmodel_7.save('CNN_7_Layers.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss =history.history['loss']
val_loss = history.history['val_loss']

EPOCHS = len(acc)
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), acc, label = 'training accuracy')
plt.plot(range(EPOCHS), val_acc, label='validaton accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS), loss, label = 'training loss')
plt.plot(range(EPOCHS), val_loss, label='validaton loss')
plt.legend(loc='upper right')
plt.title('Training and Validation loss')

best = load_model(filepath)

test_accuracy = best.evaluate(test_generator)

