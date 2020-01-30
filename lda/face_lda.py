#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:22:13 2019

@author: tushar
"""

import numpy as np
import imageio
import matplotlib.pyplot as plt


imagelist=['amber1.png','amber2.png','amy1.png','amy2.png','andrew1.png','andrew2.png','andy1.png','andy2.png','erin1.png','erin2.png','gabe1.png','gabe2.png','hill2.png','hill4.png','jack1.png','jack2.png','zach1.png','zach2.png']

train_img=[]
for file in imagelist:
    first=np.array(imageio.imread('/home/tushar/Documents/pythonProjects/SOC/lda/images/Train/'+file))
    m,n=first.shape
    first=np.reshape(first,(m*n,1))
    train_img.append(first)

pixelsize=m*n
print(pixelsize)
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

##calculating average face vector for each class
classavgface=np.ones((pixelsize,9))
for j in range(0,9):
    for i in range(0,pixelsize):
        classavgface[i][j]=(int)((train_img[i,2*j]/2+train_img[i,2*j+1]/2))
transclassavg=np.transpose(classavgface)
storedtransclassavg=transclassavg
    
##subtracting average face vector from all faces
for i in range(0,n):
    img_train[i]=img_train[i]-np.transpose(avgface)
print("img_train.shape=",img_train.shape)

##subtracting average class face vector from each of the vector
j=0
for i in range(0,n):
    img_train[i]=img_train[i]-transclassavg[j]
    if(i%2==0 and i!=0):
        j+=1
print(img_train)

##subtracting total average face vector from class average face vector
for i in range(0,9):
    transclassavg[i]=transclassavg[i]-np.transpose(avgface)

##calculating between class scatter matrix
sbetweenclass=0
for i in range(0,9):
    sbetweenclass+=np.transpose(transclassavg[0])*transclassavg[0]
sbetweenclass=sbetweenclass*2

##building scatter matrices
print(np.transpose(img_train[0]))
s1=np.transpose(img_train[0])*img_train[0]+np.transpose(img_train[1])*img_train[1]
s2=np.transpose(img_train[2])*img_train[2]+np.transpose(img_train[3])*img_train[3]
s3=np.transpose(img_train[4])*img_train[4]+np.transpose(img_train[5])*img_train[5]
s4=np.transpose(img_train[6])*img_train[6]+np.transpose(img_train[7])*img_train[7]
s5=np.transpose(img_train[8])*img_train[8]+np.transpose(img_train[9])*img_train[9]
s6=np.transpose(img_train[10])*img_train[10]+np.transpose(img_train[11])*img_train[11]
s7=np.transpose(img_train[12])*img_train[12]+np.transpose(img_train[13])*img_train[13]
s8=np.transpose(img_train[14])*img_train[14]+np.transpose(img_train[15])*img_train[15]
s9=np.transpose(img_train[16])*img_train[16]+np.transpose(img_train[17])*img_train[17]
print(s8)

##calculating within class scatter matrix
swithinclass=s1+s2+s3+s4+s5+s6+s7+s8+s9

##columns of weightmatrix are eigenvectors
eigenvectors=np.linalg.inv(swithinclass)*sbetweenclass

##projecting images onto the lda space
projectedmatrix=np.transpose(eigenvectors)*classavgface


###Testing unknown face
image=imageio.imread('Test/andrew3.png')
m,n=image.shape
image=np.reshape(image,(m*n,1))
normalisedface=image-avgface
projectedtestimage=np.transpose(eigenvectors)*normalisedface