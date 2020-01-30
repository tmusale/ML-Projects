#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:29:34 2019

@author: tushar
"""
import nltk
#nltk.download('punkt')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
        
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    
    return words
        

def cal_TF_IDF():
    mails, labels = df_train['mails'], df_train['labels']
    no_of_mails = mails.shape[0]
    
    spam_mails = labels.value_counts()[1]
    ham_mails = labels.value_counts()[0]
    
    #total_mails = spam_mails + ham_mails
    
    spam_words = 0
    ham_words = 0
    tf_spam = dict()
    tf_ham = dict()
    idf_spam = dict()
    idf_ham = dict()
    
    for i in range(no_of_mails):
        message_processed = process_message(mails[i])
        count = list() #To keep track of whether the word has ocured in the message or not.
                           #For IDF
        for word in message_processed:
            if labels[i]:
                tf_spam[word] = tf_spam.get(word, 0) + 1
                spam_words += 1
            else:
                tf_ham[word] = tf_ham.get(word, 0) + 1
                ham_words += 1
            if word not in count:
                count += [word]
        
        for word in count:
            if labels[i]:
                idf_spam[word] = idf_spam.get(word, 0) + 1
            else:
                idf_ham[word] = idf_ham.get(word, 0) + 1
                
    return tf_spam, tf_ham, idf_spam, idf_ham, spam_words, ham_words

def cal_prob(tf_spam, tf_ham):
    mails, labels = df_train['mails'], df_train['labels']
    #no_of_mails = mails.shape[0]
    
    spam_mails = labels.value_counts()[1]
    ham_mails = labels.value_counts()[0]
    
    total_mails = spam_mails + ham_mails
    
    prob_spam = dict()
    prob_ham = dict()
    
    for word in tf_spam:
        prob_spam[word] = (tf_spam[word] + 1) / spam_words + len(list(tf_spam.keys()))
        
    for word in tf_ham:
        prob_ham[word] = (tf_ham[word] + 1) / ham_words + len(list(tf_ham.keys()))
        
    prob_spam_mails = spam_mails / total_mails
    prob_ham_mails = ham_mails / total_mails
    
    return prob_spam_mails, prob_ham_mails
                

def make_dataset(dictionary):
    feature = []
    #words = []
    for i in range(0, len(df_train)):
        data = []
        blob = df_train.iloc[i, 1]
        words = blob.split(" ")
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature.append(data)
        
    return feature

df = pd.read_csv("/home/tushar/Documents/pythonProjects/SOC/naive bayes/spam.csv", encoding = 'latin-1')
#print(df.head())

'''df.drop(['Unnamed: 2'], axis = 1)
df.drop(['Unnamed: 3'], axis = 1)
df.drop(['Unnamed: 4'], axis = 1)'''

df = df.drop(df.ix[:,'Unnamed: 2':'Unnamed: 4'].head(0).columns, axis=1)

df.rename(columns={'v1':'labels', 'v2':'mails',}, inplace=True)

temp={'spam':1,'ham':0}
df.labels = [temp[i] for i in df.labels]

df_train, df_test= train_test_split(df, test_size=0.3, random_state=0)


words = []
length = len(df_train)
for i in range(0, length):
    blob = df_train.iloc[i, 1]
    words += blob.split(" ")
    
'''for i in range(0, len(words)):
    if not words[i].isalpha():
        words[i] = ""'''



#list_to_remove = words
for i in range(0,len(words)):
    if not words[i].isalpha():
        words[i] = ""
    elif len(words[i]) == 1:
        words[i] = ""
    elif words[i] == "an":
        words[i] = ""
    elif words[i] == "the":
        words[i] = ""
    elif words[i] == "is":
        words[i] = ""
    elif words[i] == "or":
        words[i] = ""
    elif words[i] == "to":
        words[i] = ""
    elif words[i] == "in":
        words[i] = ""
    elif words[i] == "and":
        words[i] = ""
    elif words[i] == "for":
        words[i] = ""

dictionary = Counter(words)
del dictionary[""]
dictionary = dictionary.most_common(2000)

features = make_dataset(dictionary)
#print(len(features))
tf_spam, tf_ham, idf_spam, idf_ham, spam_words, ham_words = cal_TF_IDF()
prob_spam_mails, prob_ham_mails = cal_prob(tf_spam, tf_ham)
