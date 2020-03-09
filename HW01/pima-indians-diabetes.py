"""
% author: burson555

% linear regression model
% pima indians diabetes prediction
%	The Pima Indians diabetes data set (pima-indians-diabetes.xlsx) 
	is a data set used to diagnostically predict whether or not a patient 
	has diabetes, based on certain diagnostic measurements included in the dataset. 
	All patients here are females at least 21 years old of Pima Indian heritage. 
	The dataset consists of M = 8 attributes 
	and one target variable, Outcome (1 represents diabetes, 0 represents no diabetes). 
	The 8 attributes include Pregnancies, Glucose, BloodPressure, BMI, insulin level, age, and so on.
"""

#import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import random

# LOADING AND PROCESSING OF DATA
# READ FROM FILE AND ADD BIAS
attributes  = {"pregnancies", "glucose", "blood_pressure", "bmi", "insulin_level", "age", "attribute7", "attribute8"}
file_path   = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/HW01/pima-indians-diabetes.csv"
# file_path = ""
diabetes_raw = np.genfromtxt(file_path, delimiter=',')
N = diabetes_raw.shape[0] # N = total number of samples
diabetes_plus_bias = np.c_[np.ones((N,1)), diabetes_raw]
# SPLIT DIABETES AND NO_DIABETES GROUPS BASED ON TARGET VALUE
columnIndex = 9
target_column = diabetes_plus_bias[:,columnIndex]
sorted_diabetes = diabetes_plus_bias[target_column.argsort()[::-1]]
split_result = np.split(sorted_diabetes, np.where(np.diff(sorted_diabetes[:,9]))[0]+1)
class_diabetes = split_result[0]
class_no_diabetes = split_result[1]

# PREVIOUS AND FAILED ATTEMPT TO SPLIT OUT TARGETS
sample_index_d = random.sample(range(class_diabetes.shape[0]), 80)
sample_index_nd = random.sample(range(class_no_diabetes.shape[0]), 80)
train_set_d = class_diabetes[sample_index_d]
train_set_nd = class_no_diabetes[sample_index_nd]
np.delete(class_diabetes, sample_index_d)
np.delete(class_no_diabetes, sample_index_nd)
target_val = sorted_diabetes[:, 9].reshape(-1, 1)

# SPLIT OUT TARGETS FROM ATTRIBUTES
target_d = class_diabetes[:, 9]
class_diabetes = class_diabetes[:, 0:9]
num_d = class_diabetes.shape[0]
target_nd = class_no_diabetes[:, 9]
class_no_diabetes = class_no_diabetes[:, 0:9]
num_nd = class_no_diabetes.shape[0]

# TRY DIFFERENT SAMPLE SIZE
samples = []
results = []
for SAMPLE_SIZE in range(40, 240, 40):
    # VARIABLES FOR STATISTICS
    COUNT   = 1000
    result  = 0
    # RUN 1000 EXPERIMENTS
    for i in range(COUNT):
        # MERGE TWO SUBSETS INTO FINAL TRAINING SET 
        X_train_d, X_test_d, t_train_d, t_test_d = \
        train_test_split(class_diabetes, target_d, test_size=(num_d-SAMPLE_SIZE)/num_d, random_state=time.time_ns()%(2**32))
        X_train_nd, X_test_nd, t_train_nd, t_test_nd = \
        train_test_split(class_no_diabetes, target_nd, test_size=(num_nd-SAMPLE_SIZE)/num_nd, random_state=time.time_ns()%(2**32))
        X_train = np.concatenate((X_train_d, X_train_nd))
        X_test = np.concatenate((X_test_d, X_test_nd))
        t_train = np.concatenate((t_train_d, t_train_nd)).reshape(-1, 1)
        t_test = np.concatenate((t_test_d, t_test_nd)).reshape(-1, 1)
        # CALCULATE, ACCELERATED BY REPLACING TENSORFLOW WITH NUMPY
        temp = X_train.transpose()
        w_val = np.linalg.inv(temp.dot(X_train)).dot(temp).dot(t_train)
        y_test_val = np.rint(X_test.dot(w_val))
#        ######################################################################
#        # JUST FOUND OUT WE DON'T HAVE TO USE TENSORFLOW
#        # numpy CAN HANDLE EVERYTHING, AND IN A EVEN FASTER MANNER
#        # DEFINE RULE AND VARIABLES FOR COMPUTATION        
#        # num_train	= number of training samples
#        # num_test  = number of test samples
#        # m         = number of attributes
#        num_train,  m 	= X_train.shape
#        num_test       = X_test.shape[0]
#        # VARIABLES WITHIN THE FORMULA
#        X   = tf.placeholder(tf.float64, shape = (None, m), name = "X")
#        t   = tf.placeholder(tf.float64, shape = (None, 1), name = "t")
#        n   = tf.placeholder(tf.float64, name = "n")
#        XT  = tf.transpose(X)
#        w   = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), t)
#        # PREDICTED VALUE
#        y_train   = tf.round(tf.matmul(X, w))
#        # VARIABLES FOR THE TEST SET
#        w_star  = tf.placeholder(tf.float64, shape = (m, 1), name = "w_star")
#        y_test  = tf.round(tf.matmul(X, w_star))
#        # RUN THE MODEL
#        with tf.Session() as sess:
#            	y_train_val, w_val = \
#            	sess.run([y_train, w], feed_dict={X: X_train, t: t_train, n:num_train})
#            	y_test_val,        = \
#            	sess.run([y_test], feed_dict={X: X_test, t: t_test, n:num_test, w_star:w_val})
#        ######################################################################
        num_test       = X_test.shape[0]
        num_match = np.count_nonzero(np.equal(y_test_val, t_test))
        result  = result + num_match/num_test
    # RETURN THE AVERAGE RESULT OF THE 1000 EXPERIMENTS
    result_averaged = result/COUNT
    results.append(result_averaged)
    samples.append(SAMPLE_SIZE)
    print("The prediction accuracy rate on %d independent experiments is %.4f" % (COUNT, result_averaged))
    print("TRAINING SIZE: %d\n" % (SAMPLE_SIZE*2))
    
# PLOTTING
plt.plot(samples, results)
plt.xlabel('x - SAMPLE_SIZE') 
plt.ylabel('y - ACCURACY')
plt.title('PIMA INDIANS DIABETES')
plt.show() 