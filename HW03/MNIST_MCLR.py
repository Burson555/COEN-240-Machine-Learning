#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:12:27 2020

@author: burson
"""

import tensorflow as tf
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

mnist = tf.keras.datasets.mnist

# DATASET ACQUISITION
(x_traino, y_train),(x_testo, y_test) = mnist.load_data()
x_train = np.reshape(x_traino,(60000,28*28))
x_test = np.reshape(x_testo,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0

# MODEL CREATION
logreg = LogisticRegression(solver='saga', multi_class='multinomial', max_iter = 100, verbose=2)

# DATA CHECKING
#import matplotlib.pyplot as plt
#plt.figure(figsize=(20,4))
#for index, (image, label) in enumerate(zip(x_train[30:35], y_train[30:35])):
#    plt.subplot(1, 5, index + 1)
#    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
#    plt.title('Training: %i\n' % label, fontsize = 20)

# MODEL FITTING
logreg.fit(x_train, y_train)

# MODEL EVALUATION
time.sleep(0.2)
predictions = logreg.predict(x_test).reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
num_test    = x_test.shape[0]
num_match   = np.count_nonzero(np.equal(predictions, y_test))
score       = num_match/num_test
print("\n\nRECOGNITION ACCURACY RATE: %.4f" % (score))

print("THE CONFUSION MATRIX: ")
cm = confusion_matrix(y_test, predictions)
print(cm)