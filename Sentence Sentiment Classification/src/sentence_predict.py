# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 17:49:31 2018

@author: pc
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding, Input,InputLayer,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU, Conv1D, MaxPool1D, Activation
from keras.models import Model,load_model
from keras.callbacks import History ,ModelCheckpoint
from keras import backend as K
import pandas as pd
import gensim
import random
import sys
import os
import re
import pickle

test_data_path = '../data/testing_data.csv'

prediction_path = '../result/'

with open(test_data_path,encoding="utf-8") as f:
    test = f.readlines()


stemmer = gensim.parsing.porter.PorterStemmer()

def preprocess(string, use_stem = True):
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")\
    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")\
    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")\
    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")\
    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")\
    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")\
    .replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return string


test_X = [preprocess("".join(sample.split(",")[1:])).strip() for sample in test[1:]]
test_X = [sent for sent in stemmer.stem_documents(test_X)]


vocab_size = None
# tokenizer = Tokenizer(num_words=vocab_size, filters="\n\t")
with open('../data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# tokenizer.fit_on_texts(train_X_cleaned + test_X + train_nolab_clean)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length = 39
sequences_test = tokenizer.texts_to_sequences(test_X)
test_X_num = pad_sequences(sequences_test, maxlen=max_length)

'''
embedding_matrix = np.zeros((len(word_index), len(word_index)))
embedding_layer = Embedding(len(word_index),output_dim= len(word_index),
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm1 = Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True,
                           kernel_initializer='he_uniform'))(embedded_sequences)
lstm2 = Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False,
                           kernel_initializer='he_uniform'))(lstm1)
bn1 = BatchNormalization()(lstm2)
dense1 = Dense(64, activation="sigmoid")(bn1)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(32, activation="sigmoid")(dropout1)
dropout2 = Dropout(0.5)(dense2)
preds = Dense(2, activation='softmax')(dropout2)
model = Model(sequence_input, preds)
model.load_weights("../model/modelW2V-00005-0.82020.h5")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
'''
'''
# ensemble

model1 = load_model("")
model2 = load_model("")
model3 = load_model("")
model4 = load_model("")
model5 = load_model("")
model6 = load_model("")
model7 = load_model("")
model8 = load_model("")
model9 = load_model("")
print("models loaded")
p1 = model1.predict(test_X_num)
p2 = model2.predict(test_X_num)
p3 = model3.predict(test_X_num)
p4 = model4.predict(test_X_num)
p5 = model5.predict(test_X_num)
p6 = model6.predict(test_X_num)
p7 = model7.predict(test_X_num)
p8 = model8.predict(test_X_num)
p9 = model9.predict(test_X_num)
# store some test_X_num for self training
pred_y_prob = (p1+p2+p3+p4+p5+p6+p7+p8+p9)/10
'''

model = load_model("../model/Bidirectional_LSTM.h5")
print(model.summary())
pred_y_prob = model.predict(test_X_num)    

pred_y = np.argmax(pred_y_prob,axis=1)
pred_y = list(pred_y)
result = []
for index,value in enumerate(pred_y):
	result.append("{0},{1}\n".format(index,value))
with open('../result/sampleSubmission.csv','w+',encoding="utf8") as f:
	f.write("id,label\n")
	f.write("".join(result))
	