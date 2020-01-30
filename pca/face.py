#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:23:13 2019

@author: tushar
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt

imagelist=['amber1.png','amber2.png','amy1.png','amy2.png','andrew1.png','andrew2.png','andy1.png','andy2.png','erin1.png','erin2.png','gabe1.png','gabe2.png','hill2.png','hill4.png','jack1.png','jack2.png','zach1.png','zach2.png']

train_img=[]

for file in imagelist:
    first=np.array(imageio.imread('/home/tushar/Documents/pythonProjects/SOC/pca/images/Train/'+file))
    m,n=first.shape
    first=np.reshape(first,(m*n,1))
    train_img.append(first)
    
print(m,n)
img_train=np.matrix(np.array(train_img))
train_img=np.transpose(img_train)
print(train_img)
m,n=train_img.shape
print("train_img.shape=",m,n)

##calculating average face vector
avgface=np.ones((m,1))
for i in range(0,m):
    avgface[i]=np.mean(train_img[i])
print("The average face vector is",avgface)

for i in range(0,n):
    img_train[i]=img_train[i]-np.transpose(avgface)
print("img_train.shape=",img_train.shape)

##calculating covariance matrix
covmat=(img_train*np.transpose(img_train))
print("The covariance matrix is")
#print(covmat)

##calculating eigen values out of covariance matrix
eigen,u=np.linalg.eig(covmat)
print("U matrix is")
#print(v)
print("eigen values are")
print(eigen)


a,s,v=np.linalg.svd(covmat)

##determining value of k i.e.(no. of principal components)
summation=np.sum(s)
print(summation)
for k in range(1,len(s)):
    sigma=0
    for i in range(0,k):
        sigma+=s[i]
    sigma=sigma/summation
    if(sigma>=0.95):
        print("selected value of k is",k)
        break
    
##selecting k eigen vectors from the set of 18 vectors
eigenvectors=np.empty((k,18))
for i in range(0,k):
    temp=np.empty((1,18))
    for j in range(0,18):
        temp[0,j]=u[i,j]
    eigenvectors[i]=temp


print("selected eigen vectors are",eigenvectors)

##mapping selected eigen vectors to original image size in training set
mappedeigenvectors=train_img*np.transpose(eigenvectors)
print(mappedeigenvectors.shape)

#temp=np.transpose(mappedeigenvectors)*mappedeigenvectors
#tempinv=np.linalg.inv(temp)
#weightmatrix=tempinv*np.transpose(mappedeigenvectors)*np.transpose(img_train)
#print(weightmatrix.shape)
weightmatrix=np.transpose(mappedeigenvectors)*np.transpose(img_train)
print(weightmatrix.shape)


###Testing unknown face
image=imageio.imread('/home/tushar/Documents/pythonProjects/SOC/pca/images/Test/andrew3.png')
m,n=image.shape
image=np.reshape(image,(m*n,1))
normalisedface=image-avgface
print(normalisedface.shape)

#temp=np.transpose(mappedeigenvectors)*mappedeigenvectors
#tempinv=np.linalg.inv(temp)
testweight=np.transpose(mappedeigenvectors)*normalisedface
print(testweight.shape)
transposeweightmatrix=np.transpose(weightmatrix)
m,n=transposeweightmatrix.shape
flag=0
store=[]
for i in range(0,m):
    difference=transposeweightmatrix[i]-np.transpose(testweight)
    totaldiff=np.sum(np.power(difference,2))
    totaldiff=np.sqrt(totaldiff)
    print(totaldiff)
    store.append(totaldiff)
a=min(store)
print()
print(a)
