#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:26:58 2019

@author: tushar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio
from skimage.io import imshow,imread

imagelist=['s101.pgm','s102.pgm','s103.pgm','s104.pgm','s105.pgm','s201.pgm','s202.pgm','s203.pgm','s204.pgm','s205.pgm','s301.pgm','s302.pgm','s303.pgm','s304.pgm','s305.pgm','s401.pgm', 's402.pgm', 's403.pgm', 's404.pgm', 's405.pgm', 's501.pgm', 's502.pgm', 's503.pgm', 's504.pgm', 's505.pgm', 's601.pgm', 's602.pgm', 's603.pgm', 's604.pgm', 's605.pgm', 's701.pgm', 's702.pgm', 's703.pgm', 's704.pgm', 's705.pgm', 's801.pgm', 's802.pgm', 's803.pgm', 's804.pgm', 's805.pgm', 's901.pgm', 's902.pgm', 's903.pgm', 's904.pgm', 's905.pgm', 's1001.pgm', 's1002.pgm', 's1003.pgm', 's1004.pgm', 's1005.pgm']

face_db = []

for file in imagelist:
    temp=np.array(imageio.imread('/home/tushar/Documents/pythonProjects/SOC/pca/images/train1/'+file))
    m,n=temp.shape
    temp=np.reshape(temp,(m*n,1))
    face_db.append(temp)

#print(m,n)
face_db=np.matrix(np.array(face_db))
face_db=np.transpose(face_db)
print(face_db)
m,n=face_db.shape
print("face_db.shape=",m,n)

mean_face=np.ones((m,1))
for i in range(0,m):
    mean_face[i]=np.mean(face_db[i])
print("The mean face vector is", mean_face)


for i in range(0,n):
    face_db = face_db[i] - (mean_face)
#print("face_db.shape=",face_db.shape)

surrogate_cov = (np.transpose(face_db) * face_db)
#print("The surrogate covariance matrix is")
#print(surrogate_cov)

eigen_values, eigen_vector = np.linalg.eig(surrogate_cov)
#print("eigen vector matrix is")
#print(eigen_vector)
print("eigen values are")
print(eigen_values)

a,s,v=np.linalg.svd(surrogate_cov)

s.sort()
sum_total = np.sum(s)

'''k = 0;
flag = 1;

while flag and k<len(s):
    k+=1 
    sum_k=0
    for i in range(k):
        sum_k+=s[i]  
    if sum_k/sum_total >= 0.95:
        flag=0'''
    
        
for k in range(1,len(s)):
    sum_k=0
    for i in range(0,k):
        sum_k+=s[i]
    temp=sum_k/sum_total
    if(temp>=0.95):
        print("selected value of k is",k)
        break

print('number of selected eigenvectors to extract k direction = ')
print(k)

feature_vector = np.empty((k,50))
for i in range(0,k):
    temp=np.empty((1,50))
    for j in range(0,50):
        temp[0,j]=eigen_vector[i,j]
    feature_vector[i] = temp
    
print("feature vector is : ")
print(feature_vector)

eigenfaces = (feature_vector) * np.transpose(face_db)
print(eigenfaces.shape)

sinature_face = eigenfaces * face_db
print(sinature_face.shape)


#testing

I = imageio.imread('/home/tushar/Documents/pythonProjects/SOC/pca/images/test1/s1001.pgm')
m, n = I.shape
I = np.reshape(I,(m*n,1))
I2 = I - mean_face
print(I2)
print(I2.shape)

projected_test_face = eigenfaces * I2
print(projected_test_face.shape)

#signature_T = np.transpose(sinature_face)
m, n = sinature_face.shape

flag = 0
distances = []

for i in range(0,m):
    difference=sinature_face[:, i] - projected_test_face
    diff_sqr = np.power(difference, 2)
    totaldiff=np.sum(diff_sqr)
    totaldiff_sqrt=np.sqrt(totaldiff)
    print(totaldiff)
    distances.append(totaldiff_sqrt)

match=min(distances)

print()
print(match)
