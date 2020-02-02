#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:41:40 2020

@author: Burson
"""

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

mnist = tf.keras.datasets.mnist

# DATASET ACQUISITION
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28*28))
x_test  = x_test.reshape((10000, 28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0

# MODEL CREATION
model = Sequential()
model.add(Dense(512, activation="relu", input_dim=28*28))
model.add(Dense(10, activation="softmax"))
model.summary()

# MODEL COMPILATION
model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# MODEL FITTING
model.fit(x_train, y_train, epochs=5, batch_size=50, verbose=2)
test_loss, test_acc = model.evaluate(x_train, y_train)
print("\n\nTRAINING SET ACCURACY RATE: %.4f" % (test_acc))

# MODEL EVALUATION
predictions_mat = model.predict(x_test)
predictions     = np.argmax(predictions_mat, axis=1)
num_test        = x_test.shape[0]
num_match       = np.count_nonzero(np.equal(predictions, y_test))
score           = num_match/num_test
print("\n\nRECOGNITION ACCURACY RATE: %.4f" % (score))
print("THE CONFUSION MATRIX: ")
cm = confusion_matrix(y_test, predictions)
print(cm)