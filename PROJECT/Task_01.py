# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""

# Import library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models,utils
from sklearn.metrics import confusion_matrix

fashion_mnist = datasets.fashion_mnist

# Data preprocessing
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)) 
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0

# One-hot encoding
train_labels = utils.to_categorical(train_labels, num_classes=10)

# Build Model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=15, verbose=2)

# Prediction
predictions = model.predict_classes(test_images)
correct_cnt = np.sum(np.array(test_labels) == np.array(predictions))
accuracy = correct_cnt/len(test_labels)

# Confusion Matrix
cm = confusion_matrix(test_labels, predictions)

# Plot confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, square = True);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy: {0:.4f}'.format(accuracy)
plt.title(all_sample_title, size = 15);


