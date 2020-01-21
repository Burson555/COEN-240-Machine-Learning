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

def calculateCenter(X, labels):
    print("haha")
    return None

def finMinDistance(point, centers):
    index = 0
    min_distance = sys.float_info.max
    for i in centers:
        
        if ()
        
def calculateDistance(point, target):
    pass
    
    
def assignCenter(X, centers):
    new_X = []
    for i in centers:
        new_X.append([])
    for j in X:
        for i in j:
            index, distace = finMinDistance(point, centers)
            new_X[index].append(i)
    return new_X

def initializeCenter(X, centers):
    for center in centers:
        for i in range(X.shape[1]):
            attr_val = X[:, i]
            center[i] = random.uniform(np.amin(attr_val), np.amax(attr_val))

# READ FROM ORIGINAL XLS FILE INTO NUMPY ARRAY
file_path_xls = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW02/Iris.xls"
file_path_csv = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW02/Iris.csv"
iris_xls = pd.read_excel(file_path_xls)
iris_xls.to_csv(file_path_csv, index = None, header=False)
iris_raw = np.genfromtxt(file_path_csv, delimiter=',')[:, 1:]
del(iris_xls)
num_column  = iris_raw.shape[1]
num_row     = iris_raw.shape[0]
X = iris_raw[:, :num_column-1]
t = iris_raw[:, num_column-1].reshape(num_row,1)

# DEFINE HYPERPARAMETERS AND INITIALIZE CENTERS
EPSILON = 10**(-5)
NUM_CENTER = 3
NUM_ITERATION = 6
centers = []
for i in range(NUM_CENTER):
    center = np.zeros((X.shape[1], 1))
    centers.append(center)
initializeCenter(X, centers)

# ALTERNATES BETWEEN ASSIGNMENT AND CLUSTER-CENTER UPDATE
for ITERATION in range(1, NUM_ITERATION+1):
    assignCenter(X, centers)
    calculateCenter(X, centers)