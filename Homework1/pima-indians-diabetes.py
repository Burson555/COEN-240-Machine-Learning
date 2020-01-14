"""
% auther: burson555

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

import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split


# LOAD DATA
attributes  = {"pregnancies", "glucose", "blood_pressure", "bmi", "insulin_level", "age", "attribute7", "attribute8"}
file_path   = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/Homework1/pima-indians-diabetes.csv"
# file_path = ""
diabetes_raw = np.genfromtxt(file_path, delimiter=',')
# N = total number of samples
N = diabetes_raw.shape[0]
diabetes_plus_bias = np.c_[np.ones((N,1)), diabetes_raw]
columnIndex = 9
target_column = diabetes_plus_bias[:,columnIndex]
sorted_diabetes = diabetes_plus_bias[target_column.argsort()[::-1]]
split_result = np.split(sorted_diabetes, np.where(np.diff(sorted_diabetes[:,9]))[0]+1)
class_diabetes = split_result[0]
class_no_diabetes = split_result[1]

# HERE I SHOULD PUT A LOOP
SAMPLE_SIZE = 200

#sample_index_d = random.sample(range(class_diabetes.shape[0]), SAMPLE_SIZE)
#sample_index_nd = random.sample(range(class_no_diabetes.shape[0]), SAMPLE_SIZE)
#train_set_d = class_diabetes[sample_index_d]
#train_set_nd = class_no_diabetes[sample_index_nd]
#np.delete(class_diabetes, sample_index_d)
#np.delete(class_no_diabetes, sample_index_nd)
#
#target_val = sorted_diabetes[:, 9].reshape(-1, 1)

target_d = class_diabetes[:, 9]
class_diabetes = class_diabetes[:, 0:9]
total_d = class_diabetes.shape[0]
target_nd = class_no_diabetes[:, 9]
class_no_diabetes = class_no_diabetes[:, 0:9]
total_nd = class_no_diabetes.shape[0]

# DEFINE RULES FOR MODEL
X_train_d, X_test_d, t_train_d, t_test_d = \
train_test_split(class_diabetes, target_d, test_size=(total_d-SAMPLE_SIZE)/total_d, random_state=time.time_ns()%(2**32))
X_train_nd, X_test_nd, t_train_nd, t_test_nd = \
train_test_split(class_no_diabetes, target_nd, test_size=(total_nd-SAMPLE_SIZE)/total_nd, random_state=time.time_ns()%(2**32))
X_train = np.concatenate((X_train_d, X_train_nd))
X_test = np.concatenate((X_test_d, X_test_nd))
t_train = np.concatenate((t_train_d, t_train_nd)).reshape(-1, 1)
t_test = np.concatenate((t_test_d, t_test_nd)).reshape(-1, 1)

# n_train	= number of training samples
# n_test    	= number of test samples
# m 		= number of attributes
n_train,m 	= X_train.shape
n_test,m 	= X_test.shape

# define the tensors
X   = tf.placeholder(tf.float64, shape = (None, m), name = "X")
t   = tf.placeholder(tf.float64, shape = (None, 1), name = "t")
n   = tf.placeholder(tf.float64, name = "n")
XT  = tf.transpose(X)
w   = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), t)
# predicted value
y   = tf.round(tf.matmul(X, w))


# mean-squared error of the prediction for TRAINING set
MSE     = tf.div(tf.matmul(tf.transpose(y-t), y-t), n)
w_star  = tf.placeholder(tf.float64, shape = (m, 1), name = "w_star")
y_test  = tf.round(tf.matmul(X, w_star))

# mean-squared error of the prediction for TEST set
MSE_test = tf.div(tf.matmul(tf.transpose(y_test-t), y_test-t), n)


# RUN THE MODEL
count   = 0
result  = 0
for i in range(100):
    with tf.Session() as sess:
    	MSE_train_val, w_val = \
    	sess.run([MSE, w], feed_dict={X: X_train, t: t_train, n:n_train})
    	MSE_test_val = \
    	sess.run([MSE_test], feed_dict={X: X_test, t: t_test, n:n_test, w_star:w_val})
    result  = result + MSE_test_val[0][0]
    count   = count + 1
    
print("The prediction accuracy rate on %d independent experiments is %.4f" % (count, result/count))