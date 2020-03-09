#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:43:53 2020

@author: Burson
"""

# ENVIRONMENT SETTING & HYPER-PARAMETER DEFINITION

from myKNN import initializeCenter, assignCenter, calculateCenter

import sys
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

NUM_CENTER = 10
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
for i in range(1, 11):
    target.append(i*np.ones(10))
target = np.array(target)
del(filename, filename_list, i)

# for d in d_list:
d = 10
# MODEL CREATION
pca = PCA(n_components=d)

# MODEL FITTING
pca_operator = pca.fit(images) # training data set, each row is one training image 
pca_projection = pca_operator.transform(images) # projection of the inputs to lower dimensional subspace

# # MODEL EVALUATION: KNN
# centers = initializeCenter(pca_projection, NUM_CENTER)
# pca_projection = np.array([pca_projection])
# M_list = []
# M, EPSILON = sys.float_info.max, 10**(-5)
# num_iteration = 0
# while(True):
#     num_iteration = num_iteration+1
#     pca_projection, new_M = assignCenter(pca_projection, centers)
#     if (M - new_M < EPSILON):
#         break
#     M = new_M
#     M_list.append(M)
#     print("ITERATION #%d\t%.4f" % (num_iteration, M))
#     centers = calculateCenter(pca_projection, centers)
# del(M, new_M)


# # input of lda is the reduced-dim data from pca:
# lda = LDA(n_components=d) # FLD /LDA
# lda_operator = lda.fit(image_list)
# train_proj_lda = lda_operator.transform(image_list).transpose() # columns are examples