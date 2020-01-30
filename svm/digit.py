#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:21:56 2019

@author: tushar
"""

from sklearn import svm, metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import math
import random
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original', data_home='./')
#mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
print(mnist.keys())
images = mnist.data
target = mnist.target
print(images)
print(target)
print(images.shape)
# shape is 70000,784
print(target.shape)
# shape is 70000
# each 784 presents image pixel values in a single row
row, col = images.shape
# we convert each 784 entry rows into a row col 2D matrix of 28* 28 to visualize the image
print(col)
a = np.zeros((int(math.pow(col, 0.5)), int(math.pow(col, 0.5))), np.int32)
r, c = a.shape
# randomly selecting a row for visualization
rand = random.randint(0, row)
img = images[rand]
test = target[rand]
#creating 2d image from pixel values
for i in range(r):
    for j in range(c):
        a[i][j] = images[rand][i * c + j]

plt.imshow(a)
plt.show()
print(target[rand])
# normalizing the complete dataset
X_data = images / 255.0
Y = target
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.5, random_state=42)
# rvf svc kernel
#uncomment to reduce the train data and test data size
#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= 0.5,random_state=42)
#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= 0.5,random_state=42)
#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= 0.5,random_state=42)
#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= 0.5,random_state=42)
#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size= 0.5,random_state=42)
param_C = 5
param_gamma = 0.05
classifier = svm.SVC(kernel='rbf', C=param_C,gamma=param_gamma)
classifier.fit(X_train, y_train)
expected = y_test
# prediction
predicted = classifier.predict(X_test)
# classification report for different classes
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
# confusion matrix
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)
# accuracy calculation
print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))