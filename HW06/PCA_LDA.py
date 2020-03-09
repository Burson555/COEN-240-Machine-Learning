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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

SPLIT_RATE = 2
CLASS_SIZE = 10
NUM_ITERATION = 20
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

# EXPERIMENTS ON d
pca_accuracy = []
for i in range(len(d_list)):
    pca_accuracy.append(0)
for iteration in range(NUM_ITERATION):
    # train/test set creation
    test_index = []
    for i in range(CLASS_SIZE):
        random_index = random.sample(range(CLASS_SIZE), SPLIT_RATE)
        for j in random_index:
            test_index.append(CLASS_SIZE*i + j)
    test_images = images[test_index]
    test_target = target[test_index]
    train_images = np.delete(images, test_index, axis=0)
    train_target = np.delete(target, test_index, axis=0)
    # PCA
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
del(i, j, d, iteration, temp, nearest_neighbor, num_match)
del(test_images, test_target, train_images, train_target, pca_predictions)
del(random_index, test_index, train_projection, test_projection)


# # input of lda is the reduced-dim data from pca:
# lda = LDA(n_components=d) # FLD /LDA
# lda_operator = lda.fit(image_list)
# train_proj_lda = lda_operator.transform(image_list).transpose() # columns are examples