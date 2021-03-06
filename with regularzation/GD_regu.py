#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:41:52 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def GD(X, Y, w, iterations, alpha):
    cost = []
    
    for i in range(0, iterations):
        h_x = X * w.T
        error = h_x - Y
        sq_error = np.square(error)
        sum = np.sum(sq_error)
        cost.append(sum / (2 * len(X)))
        #if i > 0:
            #print("iteration=",i,"|  cost=",cost[i])
        grad = (error.T * X) / len(X)
        w = w - (alpha*(grad + t/len(X)*w))
        
    return w, cost


df = pd.read_csv('Housing Price data set.csv')
size = df['lotsize'].count()

temp={'yes':1,'no':0}
df.driveway=[temp[i] for i in df.driveway]
df.recroom=[temp[i] for i in df.recroom]
df.fullbase=[temp[i] for i in df.fullbase]
df.gashw=[temp[i] for i in df.gashw]
df.airco=[temp[i] for i in df.airco]
df.prefarea=[temp[i] for i in df.prefarea]
#df.head()

#GD for all sample data
X = df.iloc[:, 2:].values
Y = df.iloc[:,1:2].values

one = np.ones(size, dtype=int)
X = np.insert(X, 0, one, axis=1)

X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

w = np.matrix(np.zeros(12))

alpha =0.001
iterations = 10000

t = 100

parameters, cost = GD(X, Y, w, iterations, alpha)
print("parameters after gradient descent = ", parameters)

it = np.arange(0, iterations)
plt.plot(it, np.array(cost))
plt.scatter(it, np.array(cost))
plt.show()

print(X.shape)
print(parameters.shape)

y_predict1 = np.matmul(X, parameters.transpose())
error1 = y_predict1 - Y
error1 = np.square(error1)
print("total squared error = ")
error1 = np.sum(error1)
print(error1/(2*len(y_predict1)))


#GD for 70% training set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print(X_test.shape)
print(Y_test.shape)
print(X_train.shape)
print(Y_train.shape)

w_train = np.matrix(np.zeros(12))

P_train, cost_train = GD(X_train, Y_train, w_train, iterations, alpha)
print("parameters after gradient descent = ", P_train)

it = np.arange(0, iterations)
plt.plot(it, np.array(cost_train))
plt.scatter(it, np.array(cost_train))
plt.show()

print(X_test.shape)
print(P_train.shape)

y_predict2 = np.matmul(X_test, P_train.transpose())
error2 = y_predict2 - Y_test
error2 = np.square(error2)
print("total squared error = ")
error2 = np.sum(error2)
print(error2/(2*len(y_predict2)))