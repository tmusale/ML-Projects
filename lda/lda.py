#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:29:57 2019

@author: tushar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import imageio
from skimage.io import imshow,imread

#PCA to reduce the dimension and then we can use the LDA on the projected faces of PCA.



imagelist=['s101.pgm','s102.pgm','s103.pgm','s104.pgm','s105.pgm','s201.pgm','s202.pgm','s203.pgm','s204.pgm','s205.pgm','s301.pgm','s302.pgm','s303.pgm','s304.pgm','s305.pgm','s401.pgm', 's402.pgm', 's403.pgm', 's404.pgm', 's405.pgm', 's501.pgm', 's502.pgm', 's503.pgm', 's504.pgm', 's505.pgm', 's601.pgm', 's602.pgm', 's603.pgm', 's604.pgm', 's605.pgm', 's701.pgm', 's702.pgm', 's703.pgm', 's704.pgm', 's705.pgm', 's801.pgm', 's802.pgm', 's803.pgm', 's804.pgm', 's805.pgm', 's901.pgm', 's902.pgm', 's903.pgm', 's904.pgm', 's905.pgm', 's1001.pgm', 's1002.pgm', 's1003.pgm', 's1004.pgm', 's1005.pgm']

face_db = []

for file in imagelist:
    temp=np.array(imageio.imread('/home/tushar/Documents/pythonProjects/SOC/lda/images/train1/'+file))
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

signature_face = eigenfaces * face_db
print(signature_face.shape)

#after PCA
PF = signature_face
m, n = PF.shape

p = 50 #number of images
num = 5 #images of each person
c = 10  #number of classes

'''for i in range(m):
    temp = []
    for j in range(c):
        for k in range(num):'''
        
#mean of each class
class_mean = np.ones((k, 10))

for j in range(0,c):
    for i in range(0,k):
        class_mean[i][j]=((PF[i,num*j]/num + PF[i,num*j+1]/num))
      
        
#mean of projected faces

mean_PF=np.ones((k, 1))

for i in range(0,m):
    mean_PF[i]=np.mean(PF[i])
print("The mean face vector is", mean_PF)
print(mean_PF.shape)    


# calculate within class scatter matrix SW
x=0
sw = []

s1 = (PF[:, 0:5] - class_mean[:, 0:1])
s1 = s1 * np.transpose(s1)

s2 = (PF[:, 5:10] - class_mean[:, 1:2])
s2 = s2 * np.transpose(s2)

s3 = (PF[:, 10:15] - class_mean[:, 2:3])
s3 = s3 * np.transpose(s3)

s4 = (PF[:, 15:20] - class_mean[:, 3:4])
s4 = s4 * np.transpose(s4)

s5 = (PF[:, 20:25] - class_mean[:, 4:5])
s5 = s5 * np.transpose(s5)

s6 = (PF[:, 25:30] - class_mean[:, 5:6])
s6 = s6 * np.transpose(s6)

s7 = (PF[:, 30:35] - class_mean[:, 6:7])
s7 = s7 * np.transpose(s7)

s8 = (PF[:, 35:40] - class_mean[:, 7:8])
s8 = s8 * np.transpose(s8)

s9 = (PF[:, 40:45] - class_mean[:, 8:9])
s9 = s9 * np.transpose(s9)

s10 = (PF[:, 45:50] - class_mean[:, 9:10])
s10 = s10 * np.transpose(s10)


SW = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10

#calculate between class scatter matrix

sb1 = class_mean[:, 0:1] - mean_PF
sb1 = sb1 * np.transpose(class_mean[:, 0:1])


sb2 = class_mean[:, 1:2] - mean_PF
sb2 = sb2 * np.transpose(class_mean[:, 1:2])


sb3 = class_mean[:, 2:3] - mean_PF
sb3 = sb3 * np.transpose(class_mean[:, 2:3])


sb4 = class_mean[:, 3:4] - mean_PF
sb4 = sb4 * np.transpose(class_mean[:, 3:4])

sb5 = class_mean[:, 4:5] - mean_PF
sb5 = sb5 * np.transpose(class_mean[:, 4:5])

sb6 = class_mean[:, 5:6] - mean_PF
sb6 = sb6 * np.transpose(class_mean[:, 5:6])

sb7 = class_mean[:, 6:7] - mean_PF
sb7 = sb7 * np.transpose(class_mean[:, 6:7])

sb8 = class_mean[:, 7:8] - mean_PF
sb8 = sb8 * np.transpose(class_mean[:, 7:8])

sb9 = class_mean[:, 8:9] - mean_PF
sb9 = sb9 * np.transpose(class_mean[:, 8:9])

sb10 = class_mean[:, 9:10] - mean_PF
sb10 = sb10 * np.transpose(class_mean[:, 9:10])

SB = sb1 + sb2 + sb3 + sb4 + sb5 + sb6 + sb7 + sb8 + sb9 + sb10

# calculating criterion function J

SWI = np.linalg.inv(SW)

J = SWI * SB

# calculate the Eigen vector and Eigen values of the Criterion function
eig_values, eig_vector = np.linalg.eig(J)

#select m best values based on the maximum Eigen values.

a,s,v=np.linalg.svd(J)

#s = sorted(s, reverse = True)
s.sort()
sum_total = np.sum(s)

for m in range(1,len(s)):
    sum_m=0
    for i in range(0,m):
        sum_m+=s[i]
    temp=sum_m/sum_total
    if(temp>=0.99):
        print("selected value of m is",m)
        break

print('number of selected eigenvectors to extract m direction = ')
print(m)

# calculating feature vector

feature = np.empty((k,m))
for i in range(0,k):
    temp=np.empty((1,m))
    for j in range(0,m):
        temp[0,j]=eig_vector[i,j]
    feature[i] = temp

print(feature.shape)
#generating fisher face FF
    
FF = np.transpose(feature) * PF

#testing

test_image = imageio.imread('/home/tushar/Documents/pythonProjects/SOC/pca/images/test1/s301.pgm')
m, n = test_image.shape
I = np.reshape(test_image,(m*n,1))
mean_img = I - mean_PF
#print(I2)
print(mean_img.shape)

# calculate projected eigen face
PEF = eigenfaces * mean_img
print(PEF.shape)

# calculate Projected Fisher Test Image

PFTI = np.transpose(feature) * PEF

#calculating distances

m, n = FF.shape

flag = 0
dist = []

for i in range(0,m):
    difference=FF[:, i] - PFTI
    diff_sqr = np.power(difference, 2)
    totaldiff=np.sum(diff_sqr)
    totaldiff_sqrt=np.sqrt(totaldiff)
    print(totaldiff_sqrt)
    dist.append(totaldiff_sqrt)

match=min(dist)

print()
print(match)



