#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:43:53 2020

@author: Burson
"""

# ENVIRONMENT SETTING & HYPER-PARAMETER DEFINITION
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import sys
sys.path.insert(1, "../HW02")
import IRIS_utils

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
projections = pca_operator.transform(images) # projection of the inputs to lower dimensional subspace

# MODEL EVALUATION





# # input of lda is the reduced-dim data from pca:
# lda = LDA(n_components=d) # FLD /LDA
# lda_operator = lda.fit(image_list)
# train_proj_lda = lda_operator.transform(image_list).transpose() # columns are examples