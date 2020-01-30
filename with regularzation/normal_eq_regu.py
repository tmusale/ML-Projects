#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:15:43 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Housing Price data set.csv')



temp={'yes':1,'no':0}
df.driveway=[temp[i] for i in df.driveway]
df.recroom=[temp[i] for i in df.recroom]
df.fullbase=[temp[i] for i in df.fullbase]
df.gashw=[temp[i] for i in df.gashw]
df.airco=[temp[i] for i in df.airco]
df.prefarea=[temp[i] for i in df.prefarea]
df.head()

#use all sample data
X = df.iloc[:, 2:].values
Y = df.iloc[:,1:2].values
size = X.shape

one = np.ones(size[0], dtype=int)
X = np.insert(X, 0, one, axis=1)

n = len(X[0])


X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

I = np.eye(n, dtype=int)
I[0][0] = 0
t = 1000

xT = np.transpose(X)
xTx = np.matmul(xT, X)
xTx = xTx + t*I
xTxI = np.linalg.inv(xTx)
xTy = np.matmul(xT, Y)
w = np.matmul(xTxI, xTy)
print(w)

y_predict1 = np.matmul(X, w)
error = y_predict1 - Y
error = np.square(error)
print("total squared error = ")
error = np.sum(error)
print(error/(2*len(y_predict1)))

'''varx = df.iloc[:, 2:3].values
print(w[0],", ", w[1])
vary = w[0] + w[1]*varx
plt.scatter(varx, Y)
plt.plot(varx, vary)'''

#use 70% for training set and 30%for test set
'''X_train = df.iloc[:400, 2:].values
Y_train = df.iloc[:400,1:2].values

X_test = df.iloc[401:, 2:].values
Y_test = df.iloc[401:,1:2].values'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


size = X_train.shape



n = len(X_train[0])

I = np.eye(n, dtype=int)
I[0][0] = 0
t = 1000

xT = np.transpose(X_train)
xTx = np.matmul(xT, X_train)
xTx = xTx + t*I
xTxI = np.linalg.inv(xTx)
xTy = np.matmul(xT, Y_train)
w_train = np.matmul(xTxI, xTy)
print(w_train)

print(X_test.shape)
print(w_train.shape)

y_predict2 = np.matmul(X_test, w_train)
error = y_predict2 - Y_test
error = np.square(error)
print("total squared error = ")
error = np.sum(error)
print(error/(2*len(y_predict2)))
