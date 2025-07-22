import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


with open('training_history_5.json', 'r') as f:
    history_loaded = json.load(f)


acc = history_loaded['accuracy']
val_acc = history_loaded['val_accuracy']
loss = history_loaded['loss']
val_loss = history_loaded['val_loss']


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()