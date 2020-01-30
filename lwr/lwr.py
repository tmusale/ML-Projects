#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:23:31 2019

@author: tushar
"""

def lwr(X,Y,gamma,w_0,w,iteration,alpha):
    cost=[]
    #xTrans=X.transpose()
    for i in range(0,iteration):
        h_x = X * w
        error = h_x - Y
        sq_error = np.square(error)
        sum = np.sum(sq_error)
        cost.append(sum / (2 * len(X)))
        #if i>0:
            #print("iteration=",i,"| cost=",cost[i])
        w_0 = w_0 - (np.sum(error) * alpha) / len(X)
        grad = (gamma.T * error) / len(X)
        w = w - (alpha*(grad))
        #w[0]=w_0
    return w, cost


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

X = (X - np.mean(X)) / np.std(X)
Y = (Y - np.mean(Y)) / np.std(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

w = np.transpose(np.matrix(np.zeros(12)))

alpha =0.001
iteration = 10000
w_0 = 0


i=100
x=[]
y=[]
count=0
for j in range(0,len(x_train)):
    if(np.array_equal(x_train[i],x_train[j])!=True and np.absolute(x_train[j][1]-x_train[i][1]) <= 0.5555):
        x.append(x_train[j])
        y.append(y_train[j])
        count=count+1
    if(count>=100):
        break
x=np.array(x)
y=np.array(y)
point=np.array(x_train[i]) #point whose prediction is to be done
gamma=[]
tau = 0.99
for i in range(0,len(x)):
    diff = x[i] - point
    sq_diff = np.square(diff)
    sq_diff = (-1) * sq_diff
    sq_diff = (sq_diff) / (2 * tau * tau)
    gamma.append(np.exp(sq_diff))
    
gamma=np.array(gamma)
parameters,cost=lwr(x,y,gamma,w_0,w,iteration,alpha)
print("parameters after lwr for prediction of 100th point:",parameters);
#y_pred=np.matmul(x_train[100],g)
#print("calculated value for given point is",y_pred)
#print("The original value for given point is",y_train[100])


    
    












