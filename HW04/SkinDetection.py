#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 08:42:29 2020

@author: Burson
"""
import math
import numpy as np
from PIL import Image
from matplotlib.pyplot import imread

# INPUT READING ------------------------------------------
train_image = imread("family.jpg")
train_truth = imread("family.png")
test_image = imread("portrait.jpg")
test_truth = imread("portrait.png")

# INPUT PROCESSING ------------------------------------------
# color code extraction AND scale invariant transformation
train_shape = train_image.shape
train_image_flat = train_image.reshape((train_shape[0]*train_shape[1], train_shape[2]))
train_image_base = np.sum(train_image_flat, axis=1) + 0.000000000000000001 # deal with 0's
train_image_flat_trans = np.transpose(train_image_flat)
train_r = np.divide(train_image_flat_trans[0], train_image_base)
train_g = np.divide(train_image_flat_trans[1], train_image_base)
# deal with 0 elements
pos_zero = np.argwhere(np.sum(train_image_flat, axis=1) == 0)
for i in pos_zero:
    train_r[int(i[0])] = 1/3
    train_g[int(i[0])] = 1/3
del(train_image, train_image_base, train_image_flat_trans, pos_zero)
# skin background split
train_label_s = train_truth.reshape((train_shape[0]*train_shape[1], train_shape[2]+1)).transpose()[1]
train_label_b = 1-train_label_s
train_r_s = train_r[np.argwhere(np.multiply(train_r, train_label_s))]
train_r_b = train_r[np.argwhere(np.multiply(train_r, train_label_b))]
train_g_s = train_g[np.argwhere(np.multiply(train_g, train_label_s))]
train_g_b = train_g[np.argwhere(np.multiply(train_g, train_label_b))]
# prior probability calculation
pp_s = np.count_nonzero(train_label_s)/train_image_flat.shape[0]
pp_b = np.count_nonzero(train_label_b)/train_image_flat.shape[0]
del(train_label_s, train_label_b)
# color code extraction AND scale invariant transformation
test_shape = test_image.shape
test_image_flat = test_image.reshape((test_shape[0]*test_shape[1], test_shape[2]))
test_image_base = np.sum(test_image_flat, axis=1) + 0.000000000000000001
test_image_flat_trans = np.transpose(test_image_flat)
test_r = np.divide(test_image_flat_trans[0], test_image_base)
test_g = np.divide(test_image_flat_trans[1], test_image_base)
# deal with 0 elements
pos_zero = np.argwhere(np.sum(test_image_flat, axis=1) == 0)
for i in pos_zero:
    test_r[int(i[0])] = 1/3
    test_g[int(i[0])] = 1/3
test_label_s = test_truth.reshape((test_shape[0]*test_shape[1], test_shape[2]+1)).transpose()[1]
del(test_image, test_image_base, test_image_flat_trans)
del(train_truth, test_truth)

# MODEL TRAINING ------------------------------------------
# use the closed-form solution we derived
miu_r_s = train_r_s.mean()
miu_g_s = train_g_s.mean()
var_r_s = train_r_s.var()
var_g_s = train_g_s.var()
miu_r_b = train_r_b.mean()
miu_g_b = train_g_b.mean()
var_r_b = train_r_b.var()
var_g_b = train_g_b.var()

# OUTPUT GENERATION ------------------------------------------
# calculate p(x|Hs) for skin
power_s_r = -np.square(test_r - miu_r_s)/(2*var_r_s)
power_s_g = -np.square(test_g - miu_g_s)/(2*var_g_s)
p_Hs_r = np.exp(power_s_r) / (math.sqrt(2*np.pi*var_r_s))
p_Hs_g = np.exp(power_s_g) / (math.sqrt(2*np.pi*var_g_s))
p_Hs = np.multiply(p_Hs_r, p_Hs_g)
# calculate p(x|Hb) for background
power_b_r = -np.square(test_r - miu_r_b)/(2*var_r_b)
power_b_g = -np.square(test_g - miu_g_b)/(2*var_g_b)
p_Hb_r = np.exp(power_b_r) / (math.sqrt(2*np.pi*var_r_b))
p_Hb_g = np.exp(power_b_g) / (math.sqrt(2*np.pi*var_g_b))
p_Hb = np.multiply(p_Hb_r, p_Hb_g)
del(test_r, test_g)
del(power_s_r, power_s_g, power_b_r, power_b_g, p_Hs_r, p_Hs_g, p_Hb_r, p_Hb_g)
# result generation applying MAP criterion
result_s = (pp_s*p_Hs - pp_b*p_Hb) > 0
# ditected binary mask generation
ones = 255*np.multiply(np.ones(test_image_flat.shape[0]), result_s)
result_array = []
for i in range(3):
    result_array.append(ones)
result_array = np.array(result_array).transpose().reshape((test_shape[0], test_shape[1], test_shape[2]))
result_array = result_array.astype(np.uint8)
result_image = Image.fromarray(result_array)
result_image.save("result.png")
del(i, ones, result_image, result_array)

# OUTPUT EVALUATION ------------------------------------------
num_s = np.count_nonzero(result_s)
num_b = test_image_flat.shape[0] - num_s
true_match_s = np.count_nonzero(np.multiply(test_label_s, result_s))
true_match_b = np.count_nonzero(np.multiply(1-test_label_s, 1-result_s))
false_match_s = np.count_nonzero(np.multiply(1-test_label_s, result_s))
false_match_b = np.count_nonzero(np.multiply(test_label_s, 1-result_s))
tpr = true_match_s / num_s
tnr = true_match_b / num_b
fpr = false_match_b / num_b
fnr = false_match_s / num_s
print("\n\n")
print("TPR: %.8f\nTNR: %.8f" % (tpr, tnr))
print("FPR: %.8f\nFNR: %.8f" % (fpr, fnr))
print("\n\n")
