#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:00:21 2019

@author: tushar
"""

def GD(X, Y, w, iteration, alpha):
    cost = []
    
    for i in range(0, iteration):
        wTx = X * w.T
        wTx = -1 * wTx
        ex = np.exp(wTx)
        h = 1 + ex
        h_x = 1 / h
        y1 = np.log2(h_x)
        y0 = np.log2(1-h_x)
        y1 = (-1) * (np.matmul(Y.T, y1))
        y0 = (-1) * (np.matmul((1-Y).T, y0))
        error = h_x - Y
        #fvalue = y1 + y0
        cost.append(y1 + y0)
        #cost.append(np.sum(np.power(error, 2)) / (2 * len(X)))
        #if i > 1:
            #print("iteration=",i,"|  cost=",cost[i])
        
        grad = (error.T * X) / len(X)
        w = w - (alpha * (grad + t / len(X) * w))
        
    return w, cost
        

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#df = pd.read_('data set for two exam results.docx')
#import textract
#text = textract.process("data set for two exam results.docx")

#import docx2txt
#my_text = docx2txt.process("data set for two exam results.docx")
#print(my_text)

df = pd.read_csv("/home/tushar/Documents/soft computing/logistic/microchip", header = None)



X = df.iloc[:, 0:2].values
Y = df.iloc[:, -1:].values

x1 = np.square(X[:,0])
x2 = np.square(X[:,-1])


size = len(X)

one = np.ones(size, dtype=int)
X = np.insert(X, 0, one, axis=1)
X = np.insert(X, 3, x1, axis=1)
X = np.insert(X, 4, x2, axis=1)

#X = (X - np.mean(X)) / np.std(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)

#w = np.matrix(np.zeros(5))
w = np.matrix(np.random.rand(X_train.shape[1],1))
w = w.T


alpha =0.1
iteration = 10000

t = 0.1

parameters, cost = GD(X_train, Y_train, w, iteration, alpha)
print("parameters after gradient descent = ", parameters)

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
#d0 = (d0 - np.mean(d0)) / np.std(d0)
#d1 = (d1 - np.mean(d1)) / np.std(d1)

'''plt.scatter([d0[:,0]],[d0[:,1]],c='b',label='y=0')
plt.scatter([d1[:,0]],[d1[:,1]],c='r',label='y=1')
#x1 = np.linspace(-2, 2, 0.1)
#x2=-(parameters[0,0]+parameters[0,1]*x1)/parameters[0,2]
#plt.plot(x1,x2,c='k',label='decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()'''

x = np.linspace(-1, 1.0, 50)
y = np.linspace(-1, 1.0, 50)
z = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        z[i, j] = parameters[0,0] + parameters[0,1]*x[i] + parameters[0,2]*y[j] + parameters[0,3]*x[i]*x[i] + parameters[0,4]*y[j]*y[j]
        
failed = plt.scatter([d0[:,0]],[d0[:,1]],c='b',label='y=0')
passed = plt.scatter([d1[:,0]],[d1[:,1]],c='r',label='y=1')
plt.contour(x, y, z, 0)
plt.xlabel('X1')
plt.ylabel('x2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()

count = 0
for i in range(0,len(y_predict)):
    if(y_predict[i] == Y_test[i]):
        count = count+1
print("predictions out of 30 test points is ",count)
accuracy =  (count / len(y_predict)) * 100
print("Accuracy achieved=", accuracy, "%")