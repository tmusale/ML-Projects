#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 12:33:33 2019

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
        w = w - (alpha*grad)
        
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
df.head()

X = df.iloc[:, 2:].values
Y = df.iloc[:,1:2].values

one = np.ones(size, dtype=int)
X = np.insert(X, 0, one, axis=1)

X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

w = np.matrix(np.zeros(12))

alpha =0.01
iterations = 10000

parameters, cost = GD(X, Y, w, iterations, alpha)
print("parameters after gradient descent = ", parameters)

it = np.arange(0, iterations)
plt.plot(it, np.array(cost))
plt.scatter(it, np.array(cost))
plt.show()

y_predict1 = np.matmul(X, parameters.transpose())
error1 = y_predict1 - Y
error1 = np.square(error1)
print("total squared error = ")
error1 = np.sum(error1)
print(error1/(2*len(y_predict1)))

