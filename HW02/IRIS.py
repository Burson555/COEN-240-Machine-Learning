"""
% author: burson555
% K-MEANS CLUSTERING
%	Iris.xls contains 150 data samples of three Iris categories, labeled by outcome values 0, 1, and 2. 
    Each data sample has four attributes: sepal length, sepal width, petal length, and petal width.

    Implement the K-means clustering algorithm to group the samples into K=3 clusters. 
    Randomly choose three samples as the initial cluster centers.
	
    Exit the iterations if the following criterion is met: 𝐽(Iter − 1) − 𝐽(Iter) < ε, 
    where ε = 10−5, and Iter is the iteration number.
    
    Plot the objective function value J versus the iteration number Iter. 
    Comment on the result. Attach the code at the end of the homework.
"""

import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt

#CHECKED
def initializeCenter(X, centers):
    attr_max = np.zeros((NUM_COLUMN, 1))
    attr_min = np.zeros((NUM_COLUMN, 1))
    for i in range(NUM_COLUMN):
        attr_val = X[:, i]
        attr_max[i] = np.amax(attr_val)
        attr_min[i] = np.amin(attr_val)
    for center in centers:
        for i in range(X.shape[1]):
            center[i][0] = random.uniform(attr_min[i], attr_max[i])
    
# X is a list of lists, where each component list represents a cluster
def assignCenter(X, centers):
    new_X = []
    for i in range(NUM_CENTER):
        new_X.append([])
    内啥 = 0
    for cluster in X:
        for point in cluster:
            point = point.reshape(4,1)
            index, min_distance = findCorresCenter(point, centers)
            new_X[index].append(point)
            内啥 = 内啥 + min_distance
    return new_X, 内啥

def findCorresCenter(point, centers):
    index = 0
    min_distance = np.linalg.norm(point-centers[0])**2
    for i in range(len(centers)):
        temp = np.linalg.norm(point-centers[i])**2
        if (temp < min_distance):
            temp = min_distance
            index = i
    return index, min_distance

def calculateCenter(X, centers):
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
#        print(cluster)
#        print(cluster_center)
    return new_centers

# READ FROM ORIGINAL XLS FILE INTO NUMPY ARRAY
file_path_xls = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW02/Iris.xls"
file_path_csv = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW02/Iris.csv"
iris_xls = pd.read_excel(file_path_xls)
iris_xls.to_csv(file_path_csv, index = None, header=False)
iris_raw = np.genfromtxt(file_path_csv, delimiter=',')[:, 1:]
del(iris_xls)
NUM_ROW     = iris_raw.shape[0]
NUM_COLUMN  = iris_raw.shape[1]-1
X = iris_raw[:, :NUM_COLUMN]
t = iris_raw[:, NUM_COLUMN].reshape(NUM_ROW,1)

# DEFINE HYPERPARAMETERS AND INITIALIZE CENTERS
EPSILON = 10**(-5)
NUM_CENTER = 3
NUM_ITERATION = 15
centers = []
for i in range(NUM_CENTER):
    center = np.zeros((NUM_COLUMN, 1))
    centers.append(center)
initializeCenter(X, centers)
#print(centers)
X = np.array([X])

# ALTERNATES BETWEEN ASSIGNMENT AND CLUSTER-CENTER UPDATE
for ITERATION in range(1, NUM_ITERATION+1):
    X, 内啥 = assignCenter(X, centers) # THE FORM OF X
#    print(内啥)
    centers = calculateCenter(X, centers)
#    print(centers)
          
print("DONE")