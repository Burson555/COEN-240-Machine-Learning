"""
% author: burson555
% K-MEANS CLUSTERING
%	Iris.xls contains 150 data samples of three Iris categories, labeled by outcome values 0, 1, and 2. 
    Each data sample has four attributes: sepal length, sepal width, petal length, and petal width.

    Implement the K-means clustering algorithm to group the samples into K=3 clusters. 
    Randomly choose three samples as the initial cluster centers.
	
    Exit the iterations if the following criterion is met: J(Iter − 1) − J(Iter) < ε, 
    where ε = 10^−5, and Iter is the iteration number.
    
    Plot the objective function value J versus the iteration number Iter. 
    Comment on the result. Attach the code at the end of the homework.
"""
# ENVIRONMENT SETTING
from IRIS_utils import *
import sys
import pandas as pd
import matplotlib.pyplot as plt

# READ FROM ORIGINAL XLS FILE INTO NUMPY ARRAY
file_path_xls = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW02/Iris.xls"
file_path_csv = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW02/Iris.csv"
iris_xls = pd.read_excel(file_path_xls)
iris_xls.to_csv(file_path_csv, index = None, header=False)
iris_raw = np.genfromtxt(file_path_csv, delimiter=',')[:, 1:]
del(iris_xls, file_path_xls, file_path_csv)
NUM_ROW     = iris_raw.shape[0]
NUM_COLUMN  = iris_raw.shape[1]-1
X = iris_raw[:, :NUM_COLUMN]
t = iris_raw[:, NUM_COLUMN].reshape(NUM_ROW,1)

# DEFINE HYPERPARAMETERS AND INITIALIZE CENTERS
M = sys.float_info.max
M_list = []
EPSILON = 10**(-5)
NUM_CENTER = 3
NUM_ITERATION = 0
centers = []
for i in range(NUM_CENTER):
    center = np.zeros((NUM_COLUMN, 1))
    centers.append(center)
initializeCenter(X, centers)
X = np.array([X])
del(center, i)

# ALTERNATES BETWEEN ASSIGNMENT AND CLUSTER-CENTER UPDATE
while(True):
    NUM_ITERATION = NUM_ITERATION+1
    X, new_M = assignCenter(X, centers)
    if (M - new_M < EPSILON):
        break
    M = new_M
    M_list.append(M)
    print("ITERATION #%d\t%.4f" % (NUM_ITERATION, M))
    centers = calculateCenter(X, centers)
del(M, new_M)

# PLOTTING
plt.plot(range(1, len(M_list)+1, 1), M_list)
plt.xlabel('x - ITERATION #') 
plt.ylabel('y - J')
plt.title('IRIS CLASSIFICATION')
plt.show() 

print("DONE")