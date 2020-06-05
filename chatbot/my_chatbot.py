# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:58:20 2020

@author: Harsh
"""

import pandas as pd
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

dataset = pd.read_excel("D:\DOWNLOADS\COVID-19-dataset.xlsx")
lastCol = 41
df = dataset.iloc[0:lastCol-1,:]


def make_ans_dictionary(ans_array):
    ans_dictionary = {}
    for i, item in enumerate(ans_array):
        ans_dictionary[i+1] = item
    return ans_dictionary
    
ans_dictionary = make_ans_dictionary(df["ans"].values)

def clean_text(text):
    text = text.lower()
    re.sub(r"covid-19", "covid19", text)
    re.sub(r"covid 19", "covid19", text)
    re.sub(r"novel coronavirus", "covid19", text)
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

def make_que_dictionary(que_array):
    que_dictionary = {}
    for i, item in enumerate(que_array):
        que_arr = []
        for text in item.split('%'):
            que_arr.append(clean_text(text))
        que_dictionary[i+1] = que_arr
    
    return que_dictionary

que_dictionary = make_que_dictionary(df["alternate-que"].values)

def training_dataset(que_dictionary):
    que_train = []
    y_train = []
    for i, item in que_dictionary.items():
        for que in item:
            que_train.append(que)
            y_train.append(i)
    
    return np.array(que_train).reshape(-1,1), np.array(y_train)
    
que_train, y_train = training_dataset(que_dictionary)

def word_count(train):
    word2count = {}
    for text in train:
        for word in text.split():
            if word in word2count:
                word2count[word] += 1
            else:
                word2count[word] = 1
    return word2count

word2count = word_count(que_train[:,0])

def word_to_id(word2count, threshold):
    word2int = {}
    counter = 0
    for word, count in word2count.items():
        if count > threshold:
            word2int[word] = counter
            counter += 1
    return word2int
    
word2int = word_to_id(word2count, 0)

def id_to_word(word2int):
    int2word = {}
    for word, i in word2int.items():
        int2word[i] = word
    return int2word

int2word = id_to_word(word2int)

def que_to_int(que_train, word2int):
    que_array = []
    for que in que_train[:,0]:
        ints = []
        for word in que.split():
            ints.append(word2int[word])
        que_array.append(ints)
    return np.array(que_array)

que_array = que_to_int(que_train, word2int)

print(que_array)
xdum = np.array([[1,2,3],[2,3],[5,6,43,2]])
ydum = np.array([1,2,3])
sorted_arr = []
#for i in enumerate(que_array):
    #print(que_array[i[0]])

X_train = np.reshape(que_array, (que_array.shape[0], 1, 1))


classifier = Sequential()

classifier.add(LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1],1)))
classifier.add(LSTM(units=20, return_sequences=True))
classifier.add(LSTM(units=20))

classifier.add(Dense(units=1))

classifier.compile(optimizer='adam', loss='mean_squared_error')

classifier.fit(X_train, y_train, epochs =50, batch_size=5)

# Error in classifier bcoz train data is a list and now numpy array

































