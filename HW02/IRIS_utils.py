#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:13:25 2020

@author: Burson
"""
# ENVIRONMENT SETTING & FUNCTION DEFINITION
import numpy as np
import random

def initializeCenter(X, NUM_CENTER):
    NUM_COLUMN = X.shape[1]
    centers = []
    for i in range(NUM_CENTER):
        center = np.zeros((NUM_COLUMN, 1))
        centers.append(center)
    NUM_COLUMN = centers[0].shape[0]
    attr_max = np.zeros((NUM_COLUMN, 1))
    attr_min = np.zeros((NUM_COLUMN, 1))
    for i in range(NUM_COLUMN):
        attr_val = X[:, i]
        attr_max[i] = np.amax(attr_val)
        attr_min[i] = np.amin(attr_val)
    # print(attr_max)
    # print(attr_min)
    for center in centers:
        for i in range(X.shape[1]):
            center[i][0] = random.uniform(attr_min[i], attr_max[i])
    return centers
    
# X is a list of lists, where each component list represents a cluster
def assignCenter(X, centers):
    NUM_CENTER = len(centers)
    NUM_COLUMN = centers[0].shape[0]
    new_X = []
    for i in range(NUM_CENTER):
        new_X.append([])
    M = 0
    for cluster in X:
        for point in cluster:
            point = point.reshape(NUM_COLUMN, 1)
            index, min_distance = findCorresCenter(point, centers)
            new_X[index].append(point)
            M = M + min_distance
    return new_X, M

def findCorresCenter(point, centers):
    index = 0
    min_distance = np.linalg.norm(point-centers[0])**2
    for i in range(len(centers)):
        temp = np.linalg.norm(point-centers[i])**2
        if (temp < min_distance):
            min_distance = temp
            index = i
    return index, min_distance

def calculateCenter(X, centers):
    NUM_CENTER = len(centers)
    NUM_COLUMN = centers[0].shape[0]
    new_centers = []
    for i in range(NUM_CENTER):
        cluster = X[i]
        cluster_center = np.zeros((NUM_COLUMN, 1))
        if (len(cluster) == 0):
            new_centers.append(centers[i])
            continue
        for i in range(NUM_COLUMN):
            cluster = np.array(cluster)
            cluster = cluster.reshape(cluster.shape[0], NUM_COLUMN)
            attr_val = cluster[:, i]
            cluster_center[i][0] = np.mean(attr_val)
        new_centers.append(cluster_center)
    return new_centers