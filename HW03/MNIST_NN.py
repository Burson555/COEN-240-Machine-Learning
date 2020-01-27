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
from sklearn import metrics

mnist = tf.keras.datasets.mnist

# DATASET ACQUISITION
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, (60000,28*28))
x_test = np.reshape(x_test, (10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0

# MODEL CREATION
model = Sequential()
model.add(Dense(512, input_dim=(28*28), activation="relu"))
model.add(Dense(10, activation="softmax"))

# MODEL COMPILATION
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# MODEL FITTING
model.fit(x_train, y_train, epochs=5, batch_size=50, verbose=2)

# MODEL EVALUATION
predictions = model.predict(x_test).reshape(-1, 1)
num_test    = x_test.shape[0]
num_match   = np.count_nonzero(np.equal(predictions, y_test))
score       = num_match/num_test
print(score)




