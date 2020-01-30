#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:52:28 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio
from skimage.io import imshow,imread

def cal_cov(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    s = []
    n = len(x)
    for i in range(n):
        temp1 = x[i] - x_mean
        temp2 = y[i] - y_mean
        s.append(temp1 * temp2)
        
    s = np.sum(s)
    s = s / (n-1)
    
    return s 
    

one = imageio.imread('images/1.gif')
#print(one)
two = imageio.imread('images/2.gif')
three = imageio.imread('images/3.gif')
four = imageio.imread('images/4.gif')

df1 = pd.read_csv("r.txt", header = None)
df2 = pd.read_csv("nr.txt", header = None)

Xr = df1.iloc[:,0:1].values
Yr = df1.iloc[:,1:2].values


Xnr = df2.iloc[:,0:1].values
Ynr = df2.iloc[:,1:2].values

mean_Xr = np.mean(Xr)
mean_Yr = np.mean(Yr)
mean_Xnr = np.mean(Xnr)
mean_Ynr = np.mean(Ynr)

#Calculate Mean of River Class
Rr = []
Gr = []
Br = []
Ir = []

for i in range(len(Xr)):
    Rr.append(one[Xr[i], Yr[i]])
    Gr.append(two[Xr[i], Yr[i]])
    Br.append(three[Xr[i], Yr[i]])
    Ir.append(four[Xr[i], Yr[i]])
    
Rr = np.matrix(Rr)
Gr = np.matrix(Gr)
Br = np.matrix(Br)
Ir = np.matrix(Ir)

river_mean = [np.mean(Rr), np.mean(Gr), np.mean(Br), np.mean(Ir)]
river_mean = np.matrix(river_mean)
print('mean of river class = ')
print(river_mean)

#Calculate Mean of non River Class
Rnr = []
Gnr = []
Bnr = []
Inr = []

for i in range(len(Xnr)):
    Rnr.append(one[Xnr[i], Ynr[i]])
    Gnr.append(two[Xnr[i], Ynr[i]])
    Bnr.append(three[Xnr[i], Ynr[i]])
    Inr.append(four[Xnr[i], Ynr[i]])
    
Rnr = np.matrix(Rnr)
Gnr = np.matrix(Gnr)
Bnr = np.matrix(Bnr)
Inr = np.matrix(Inr)

nonriver_mean = [np.mean(Rnr), np.mean(Gnr), np.mean(Bnr), np.mean(Inr)]
nonriver_mean = np.matrix(nonriver_mean)
print('mean of non river class = ')
print(nonriver_mean)

#Calculate the Covariance Matrix for River Class

river_cov = [[cal_cov(Rr, Rr), cal_cov(Rr, Gr), cal_cov(Rr, Br), cal_cov(Rr, Ir)],
             [cal_cov(Gr, Rr), cal_cov(Gr, Gr), cal_cov(Gr, Br), cal_cov(Gr, Ir)],
             [cal_cov(Br, Rr), cal_cov(Br, Gr), cal_cov(Br, Br), cal_cov(Br, Ir)],
             [cal_cov(Ir, Rr), cal_cov(Ir, Gr), cal_cov(Ir, Br), cal_cov(Ir, Ir)]]

river_cov = np.matrix(river_cov)
print('the Covariance Matrix for River Class')
print(river_cov)

#Calculate the Covariance Matrix for non River Class

nonriver_cov = [[cal_cov(Rnr, Rnr), cal_cov(Rnr, Gnr), cal_cov(Rnr, Bnr), cal_cov(Rnr, Inr)],
             [cal_cov(Gnr, Rnr), cal_cov(Gnr, Gnr), cal_cov(Gnr, Bnr), cal_cov(Gnr, Inr)],
             [cal_cov(Bnr, Rnr), cal_cov(Bnr, Gnr), cal_cov(Bnr, Bnr), cal_cov(Bnr, Inr)],
             [cal_cov(Inr, Rnr), cal_cov(Inr, Gnr), cal_cov(Inr, Bnr), cal_cov(Inr, Inr)]]

nonriver_cov = np.matrix(river_cov)
print('the Covariance Matrix for non River Class')
print(nonriver_cov)

#for test data

out = []
P1 = 0.4
P2 = 0.1

for i in range(512):
    temp = []
    for j in range(512):
        test = [one[i, j], two[i, j], three[i, j], four[i, j]]
        test = np.matrix(test)
        
        test_r = test - river_mean
        test_nr = test - nonriver_mean
        
        river_cov_I = np.linalg.inv(river_cov)
        nonriver_cov_I = np.linalg.inv(nonriver_cov)
        
        river_class = test_r * river_cov_I * test_r.T
        nonriver_class = test_nr * nonriver_cov_I * test_nr.T
        
        det_river_cov = np.linalg.det(river_cov)
        det_nonriver_cov = np.linalg.det(nonriver_cov)
        sqrt_det_river = np.sqrt(det_river_cov)
        sqrt_det_nonriver = np.sqrt(det_nonriver_cov)
        
        p1 = (-0.5) * (1 / sqrt_det_river) * np.exp(river_class)
        p2 = (-0.5) * (1 / sqrt_det_nonriver) * np.exp(nonriver_class)
        
        if ((P1 * p1) >= (P2 * p2)):
            temp.append(255)
        else:
            temp.append(0)
    out.append(temp)
    
output = np.uint8(np.array(out))
imageio.imwrite('images/output5.jpeg',output[:,:])



