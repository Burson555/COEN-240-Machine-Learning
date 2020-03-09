#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:43:53 2020

@author: Burson
"""

# ENVIRONMENT SETTING & HYPER-PARAMETER DEFINITION
from utils import findNearestNeighbor

import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

TEST_SIZE = 2
CLASS_SIZE = 10
# NUM_ITERATION = 1
NUM_ITERATION = 20
d_for_FLD = 40
# d_list = [1, 2, 3, 6]
# d_list = [1, 2, 3, 6, 10]
d_list = [1, 2, 3, 6, 10, 20, 30]

# DATASET ACQUISITION: input shape = (112, 92)
# images gathering
images = []
filename_list = glob.glob("att_faces_10/*/*.pgm")
for filename in filename_list:
    images.append(np.array(Image.open(filename)))
images = np.array(images)
images_shape = images.shape
images = images.reshape((images_shape[0], images_shape[1]*images_shape[2]))
images = images.astype(np.float64)
# target creation
target = []
for i in range(1, CLASS_SIZE+1):
    target.append(i*np.ones(CLASS_SIZE))
target = np.array(target)
target = target.reshape((images_shape[0], 1))
del(filename, filename_list, i)

# PCA
pca_accuracy = []
for i in range(len(d_list)):
    pca_accuracy.append(0)
for iteration in range(NUM_ITERATION):
    # train/test set creation
    test_index = []
    for i in range(CLASS_SIZE):
        random_index = random.sample(range(CLASS_SIZE), TEST_SIZE)
        for j in random_index:
            test_index.append(CLASS_SIZE*i + j)
    test_images = images[test_index]
    test_target = target[test_index]
    train_images = np.delete(images, test_index, axis=0)
    train_target = np.delete(target, test_index, axis=0)
    # experiments on d
    for i in range(len(d_list)):
        d = d_list[i]
        # MODEL CREATION
        pca = PCA(n_components=d)
        # MODEL FITTING
        # training the model, each row is one training image 
        pca_operator = pca.fit(train_images)
        # projection of the inputs to lower dimensional subspace
        train_projection = pca_operator.transform(train_images)
        test_projection = pca_operator.transform(test_images)
        # MODEL EVALUATION
        pca_predictions = []
        for temp in test_projection:
            nearest_neighbor = findNearestNeighbor(train_projection, temp)
            pca_predictions.append(train_target[nearest_neighbor])
        pca_predictions = np.array(pca_predictions)
        num_match = np.count_nonzero(np.equal(pca_predictions, test_target))
        pca_accuracy[i] = pca_accuracy[i] + num_match/test_target.shape[0]
pca_accuracy = np.array(pca_accuracy)
pca_accuracy = pca_accuracy/NUM_ITERATION
del(pca, pca_operator, pca_predictions)

# FLD
lda_accuracy = []
for i in range(len(d_list)):
    lda_accuracy.append(0)
# dimension reduction with PCA
pca = PCA(n_components=d_for_FLD)
pca_operator = pca.fit(images)
pca_projection = pca_operator.transform(images)
for iteration in range(NUM_ITERATION):
    # train/test set creation
    test_index = []
    for i in range(CLASS_SIZE):
        random_index = random.sample(range(CLASS_SIZE), TEST_SIZE)
        for j in random_index:
            test_index.append(CLASS_SIZE*i + j)
    test_images = pca_projection[test_index]
    test_target = target[test_index]
    train_images = np.delete(pca_projection, test_index, axis=0)
    train_target = np.delete(target, test_index, axis=0)
    # experiments on d
    for i in range(len(d_list)):
        d = min(d_list[i], test_images.shape[1], CLASS_SIZE-1)
        # MODEL CREATION
        lda = LDA(n_components=d)
        # MODEL FITTING
        lda_operator = lda.fit(train_images, train_target.reshape(train_target.shape[0], ))
        # projection of the inputs to lower dimensional subspace
        train_projection = lda_operator.transform(train_images)
        test_projection = lda_operator.transform(test_images)
        # MODEL EVALUATION
        lda_predictions = []
        for temp in test_projection:
            nearest_neighbor = findNearestNeighbor(train_projection, temp)
            lda_predictions.append(train_target[nearest_neighbor])
        lda_predictions = np.array(lda_predictions)
        num_match = np.count_nonzero(np.equal(lda_predictions, test_target))
        lda_accuracy[i] = lda_accuracy[i] + num_match/test_target.shape[0]
        # accuracy = num_match/test_target.shape[0]
lda_accuracy = np.array(lda_accuracy)
lda_accuracy = lda_accuracy/NUM_ITERATION
del(pca, pca_operator, pca_projection)
del(lda, lda_operator, lda_predictions)

del(i, j, d, iteration, temp, nearest_neighbor, num_match)
del(test_images, test_target, train_images, train_target)
del(random_index, test_index, train_projection, test_projection)

# PLOT
width, height, dpi = 8, 6, 128
plt.figure(figsize=(width, height), dpi=dpi)
plt.title("PCA vs LDA")
plt.xlabel("d value")
plt.ylabel("recognition accuracy")
# plt.plot(d_list, pca_accuracy, label="PCA accuracy")
plt.plot(d_list, lda_accuracy, label="LDA accuracy")
plt.legend()
plt.savefig("PCAvsLDA.png")
del(width, height, dpi)