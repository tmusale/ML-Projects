#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:24:04 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmoid(temp):
    temp = -temp
    ex_temp = np.exp(temp)
    ex_temp = 1 + ex_temp
    return (1/ex_temp)

def cal_deltaJ(X, Y, w):
    wT = np.transpose(w)
    wTx = np.matmul(X, wT)
    h_x = sigmoid(wTx)
    error = h_x - Y
    delta = (error.T * X)/len(X)
    return delta
    
def hessian(X, w):
    wT = np.transpose(w)
    wTx = np.matmul(X, wT)
    h_x = sigmoid(wTx)
    temp=[]
    for i in range(0,70):
        temp.append(0.25)
    D=np.diag(np.array(temp))
    temp = 1 - h_x
    s = np.matmul(X.T,D)
    prod = np.dot(s,X)
    
    return (prod / len(X))
    

def newton_method(X, Y, w, iteration):
    cost = []
    for i in range(0, iteration):
        wT = np.transpose(w)
        wTx = np.matmul(X, wT)
        h_x = sigmoid(wTx)
        #error = h_x - Y
        tmp = (-1) * Y.T * np.log(h_x) - (1 - Y.T) * np.log((1-h_x))
        cost.append(tmp)
        if i > 1:
            print("iteration=",i,"|  cost=",cost[i])
        #J = np.sum(tmp) / len(Y)
        #H = np.matmul((h_x * (1 - h_x).T * X),(X.T)) / len(Y)
        #delta_J = np.sum(error.T * X, axis = 1) / len(Y)
        delta_J=cal_deltaJ(X,Y,w)
        H = hessian(X, w)
        #print(H)
        HI = np.linalg.inv(H)
        temp = delta_J * HI
        w = w - temp
    
    return w, cost

df = pd.read_csv("/home/tushar/Documents/pythonProjects/SOC/logistic/abc", header = None)



X = df.iloc[:, 0:2].values
Y = df.iloc[:, -1:].values

size = len(X)

one = np.ones(size, dtype=int)
X = np.insert(X, 0, one, axis=1)

X = (X - np.mean(X)) / np.std(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

w = np.matrix(np.zeros(3))

'''X = X_train
wT = np.transpose(w)
wTx = np.matmul(X, wT)
h_x = sigmoid(wTx)
temp = 1 - h_x
xTx = np.matmul(X, X.T)
prod = h_x * temp.T * xTx
prod = np.sum(prod) / len(X)
print(prod / len(X))'''

#alpha =0.01
iteration = 20

parameters, cost  = newton_method(X_train, Y_train, w, iteration)
print("parameters after newton's method = ", parameters)

it = np.arange(0, iteration)
plt.plot(it, np.squeeze(cost))
plt.scatter(it, np.squeeze(cost))
plt.show()

y_predict = np.matmul(X_test, parameters.transpose())
y_predict = (-1) * y_predict
ex = np.exp(y_predict)
h = 1 + ex
y_predict = 1 / h
#pred = [y_predict >= 0.5]
#print(np.mean(pred == Y_test.flatten()) * 100)
y_predict = np.where(y_predict >= 0.5,1,0)
y_predict = np.squeeze(y_predict)

d0 = []
d1 = []

for i in range(len(df)):
    #temp = df[i][0]
    a = df[0][i]
    b = df[1][i]
    c = df[2][i] 
    if(c==0):
            d0.append([a,b])
    if(c==1):
            d1.append([a,b])

d0 = np.array(d0)
d1 = np.array(d1)
d0 = (d0 - np.mean(d0)) / np.std(d0)
d1 = (d1 - np.mean(d1)) / np.std(d1)

plt.scatter([d0[:,0]],[d0[:,1]],c='b',label='y=0')
plt.scatter([d1[:,0]],[d1[:,1]],c='r',label='y=1')
#x1=np.arange(-2,2, )
x = np.arange(-2, 2, 0.1)
y = -(parameters[0,0]+parameters[0,1]*x) / parameters[0,2]
#x1 = x1*x1
#x2 = x2*x2
plt.plot(x, y, label='decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

count = 0
for i in range(0,len(y_predict)):
    if(y_predict[i] == Y_test[i]):
        count = count+1
print("predictions out of 30 test points is ",count)
accuracy =  (count / len(y_predict)) * 100
print("Accuracy achieved=", accuracy, "%")