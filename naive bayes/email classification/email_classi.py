#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:55:41 2019

@author: tushar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import re
import string
import math

def clean(s):
    translator = str.maketrans("", "", string.punctuation)
    return s.translate(translator)

def tokenize(text):
    text = clean(text).lower()
    text = text.lower()
    return re.split("\W+", text)

def get_word_counts(words):
    word_counts = {}
    for word in words:
        #word_counts[word] = word_counts.get(word, 0.0) + 1.0
        c = word_counts.get(word, 0.0)
        c += 1
        word_counts[word] = c
        
    return word_counts

def prediction(X):
    result = []
    #num_mails['spam'] = Y_train.labels.value_counts()[1]
    #num_mails['ham'] = Y_train.labels.value_counts()[0]
    r = len(X)
    for i in range(len(X)):
        x = X.iloc[i,0]
        
        spam_score = 0
        ham_score = 0
        
        counts = get_word_counts(tokenize(x))
        
        for word,_ in counts.items():
            if word not in dictionary:
                continue
            
            #added laplace smoothing
            prob_spam_giv_w = math.log( (word_counts['spam'].get(word, 0.0)+1) / ( num_mails['spam'] + len(dictionary)) )
            prob_ham_giv_w = math.log( (word_counts['ham'].get(word, 0.0)+1) / ( num_mails['ham'] + len(dictionary)) )
        
            spam_score += prob_spam_giv_w
            ham_score += prob_ham_giv_w
            
        spam_score += prob['spam']
        ham_score += prob['ham']
        
        if spam_score > ham_score:
            result.append(1)
        else:
            result.append(0)
            
        #print(r)
        #r = r-1
            
    #print(result)
    return result
        

'''def fit(X, Y):
    num_mails = {}
    prob = {}
    word_counts = {}
    dictionary = {}
    
    n = len(X)
    num_mails['spam'] = Y.value_counts()[1]
    num_mails['ham'] = Y.value_counts()[0]
    
    prob['spam'] = math.log(num_mails['spam'] / n)
    prob['ham'] = math.log(num_mails['ham'] / n)
    
    word_counts['spam'] = {}
    word_counts['ham'] = {}
    
    for x, y in zip(X, Y):'''
        
    


df = pd.read_csv("/home/tushar/Documents/pythonProjects/SOC/naive bayes/email classification/spam.csv", encoding = 'latin-1')

df = df.drop(df.ix[:,'Unnamed: 2':'Unnamed: 4'].head(0).columns, axis=1)

df.rename(columns={'v1':'labels', 'v2':'mails',}, inplace=True)

temp={'spam':1,'ham':0}
df.labels = [temp[i] for i in df.labels]

X = df.iloc[:, -1:]
Y = df.iloc[:, 0:-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

#X = X_train
#Y = Y_train

#i = 0
#print(Y.iloc[0])
#print(len(X))


num_mails = {}
prob = {}
word_counts = {}
dictionary = []
    
n = len(X_train)
num_mails['spam'] = Y_train.labels.value_counts()[1]
num_mails['ham'] = Y_train.labels.value_counts()[0]
    
prob['spam'] = math.log(num_mails['spam'] / n)
prob['ham'] = math.log(num_mails['ham'] / n)
    
word_counts['spam'] = {}
word_counts['ham'] = {}
    
r = n
for i in range(len(X_train)):
    x = X_train.iloc[i,0] 
    y = Y_train.iloc[i].values
    if y == 1:
        c = 'spam'
    else:
        c = 'ham'
        
    counts = get_word_counts(tokenize(x))
    
    for word, count in counts.items():
        if word not in dictionary:
            dictionary.append(word)
        if word not in word_counts[c]:
            word_counts[c][word] = 0.0
        
        word_counts[c][word] += count
    
    #print(r)
    #r = r-1
    

predict = prediction(X_test)

count = 0
for i in range(0,len(predict)):
    if(predict[i] == Y_test.iloc[i,0]):
        count = count+1

accuracy =  (count / len(predict)) * 100
print("Accuracy achieved=", accuracy, "%")
    
    
    
    
    
    
    
    