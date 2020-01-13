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
from sklearn.model_selection import train_test_split

#from sklearn.datasets import fetch_california_housing
#
#housing = fetch_california_housing()
#N = housing.data.shape[0]
#housing_data_plus_bias = np.c_[np.ones((N,1)), housing.data]
#target_val = housing.target.reshape(-1, 1)

# load dataset
attributes = {"pregnancies", "glucose", "blood_pressure", "bmi", "insulin_level", "age", "attribute7", "attribute8"}
file_path = "/Users/bosen/Library/Mobile Documents/com~apple~CloudDocs/Portal/COEN 240/Assignment/Homework1/pima-indians-diabetes.csv"
# file_path = ""
diabetes_data = np.genfromtxt(file_path, delimiter=',')
# N = total number of samples
N = diabetes_data.shape[0]
diabetes_data_plus_bias = np.c_[np.ones((N,1)), diabetes_data[:, 0:8]]
target_val = diabetes_data[:, 8].reshape(-1, 1)

X_train, X_test, t_train, t_test = \
train_test_split(diabetes_data_plus_bias, target_val, test_size=0.2, random_state=32)

# n_train	= number of training samples
# n_test	= number of test samples
# m 		= number of attributes
n_train,m 	= X_train.shape
n_test,m 	= X_test.shape

# define the tensors
X = tf.placeholder(tf.float64, shape = (None, m), name = "X")
t = tf.placeholder(tf.float64, shape = (None, 1), name = "t")
n = tf.placeholder(tf.float64, name = "n")
XT = tf.transpose(X)
w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), t)
# predicted value
y = tf.round(tf.matmul(X, w))


# mean-squared error of the prediction for TRAINING set
MSE = tf.div(tf.matmul(tf.transpose(y-t), y-t), n)
w_star = tf.placeholder(tf.float64, shape = (m, 1), name = "w_star")
y_test = tf.round(tf.matmul(X, w_star))

# mean-squared error of the prediction for TEST set
MSE_test = tf.div(tf.matmul(tf.transpose(y_test-t), y_test-t), n)

count = 0
result = 0
for i in range(100):
    with tf.Session() as sess:
    	MSE_train_val, w_val = \
    	sess.run([MSE, w], feed_dict={X: X_train, t: t_train, n:n_train})
    	MSE_test_val = \
    	sess.run([MSE_test], feed_dict={X: X_test, t: t_test, n:n_test, w_star:w_val})
    result  = result + MSE_test_val[0][0]
    count   = count + 1
    
print("The prediction accuracy rate on %d independent experiments is %.4f" % (count, result/count))