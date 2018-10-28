# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 21:10:28 2018

@author: acer
"""

import os
import sys
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint
from keras.layers import Input,Embedding,Dense,Dropout,Flatten,Reshape,Lambda
from keras.layers.merge import concatenate,dot,add
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt

def DNN(n_users,n_movies,dim,dropout=0.1):
    u_input = Input(shape = (4,)) # use bias
	#u_input = Input(shape = (1,)) 
    u = Embedding(n_users,dim)(u_input)
    u = Flatten()(u)
    
    m_input = Input(shape = (19,)) # use bias
	#m_input = Input(shape = (1,))
    m = Embedding(n_movies,dim)(m_input)
    m = Flatten()(m)
    
    out = concatenate([u,m])
    out = Dropout(dropout)(out)
    out = Dense(256,activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(128,activation='relu')(out)
    out = Dropout(dropout)(out)
    out = Dense(64,activation='relu')(out)
    out = Dropout(0.15)(out)
    out = Dense(dim,activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(1,activation='relu')(out)
    
    model = Model(inputs=[u_input,m_input],outputs=out)
    return model

def Matrix_Factorization(n_users,n_movies,dim):
    u_input = Input(shape=(1,))
    u = Embedding(n_users,dim,embeddings_regularizer=l2(1e-5))(u_input)
    u = Reshape((dim,))(u)
    u = Dropout(0.1)(u)
    
    m_input = Input(shape=(1,))
    m = Embedding(n_movies,dim,embeddings_regularizer=l2(1e-5))(m_input)
    m = Reshape((dim,))(m)
    m = Dropout(0.1)(m)
    
    u_bias = Embedding(n_users,1,embeddings_regularizer=l2(1e-5))(u_input)
    u_bias = Reshape((1,))(u_bias)
    
    m_bias = Embedding(n_movies,1,embeddings_regularizer=l2(1e-5))(m_input)
    m_bias = Reshape((1,))(m_bias)
    
    out = dot([u,m],-1)
    out = add([out,u_bias,m_bias])
    out = Lambda(lambda x: x + K.constant(3.581712))(out)
    
    model = Model(inputs=[u_input,m_input],outputs=out)
    return model
    
def rmse(y_true,y_pred):
    y_pred = K.clip(y_pred,1.,5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def normalization(row):
    row['Rating'] = (row['Rating']-np.std(row['Rating'],axis=0))/np.mean(row['Rating'],axis=0)
    return row    

def make_users(row,matrix):
    matrix[row['UserID']] = [row['UserID'],row['Gender'],row['Age'],row['Occupation']]
    return row

def categorize_movie(row,matrix,idx_map):
    x = [0] * len(classes)
    for g in row['Genres'].split('|'):
        x[idx_map[g]] = 1
    matrix[row['movieID']] = [row['movieID']] + x

def main():
    ratings = pd.read_csv(train_path,usecols = ['UserID','MovieID','Rating'])
    max_userid = ratings['UserID'].drop_duplicates().max()
    max_movieid = ratings['MovieID'].drop_duplicates().max()
    ratings['User_emb_id'] = ratings['UserID']-1
    ratings['Movie_emb_id'] = ratings['MovieID']-1
    print('{} ratings loaded.'.format(ratings.shape[0]))

    if normalize:
        ratings = normalization(ratings)
    
    if enable_bias:
        users = pd.read_csv(users_path,sep='::',engine='python',usecols = ['UserID','Gender','Age','Occupation'])
        users['UserID'] -= 1
        users['Gender'][users['Gender']=='F'] = 0
        users['Gender'][users['Gender']=='M'] = 1 
        users_mx = {}
        users.apply(lambda x:make_users(x,users_mx),axis = 1)
        print('{} description of {} users loaded'.format(len(users), max_userid))
        
        movies = pd.read_csv(movies_path,sep='::',engine = 'python',usecols = ['movieID','Genres'])
        movies['movieID'] -=1
        movies_mx = {}
        classes_idx = {}
        for i,c in enumerate(classes):
            classes_idx[c] = i
        movies.apply(lambda x: categorize_movie(x, movies_mx, classes_idx), axis=1)
        print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))
        
    maximum = {}
    maximum['max_userid'] = [max_userid]
    maximum['max_movieid'] = [max_movieid]
    maximum['dim'] = [DIM]
    pd.DataFrame(data = maximum).to_csv(MAX_FILE,index = False)
    print('Max info save to {}'.format(MAX_FILE))

    ratings = ratings.sample(frac=1)
    Users = ratings['User_emb_id'].values
    print('Users : {}, shape = {}'.format(Users,Users.shape))
    Movies = ratings['Movie_emb_id'].values
    print('Movies : {}, shape = {}'.format(Movies,Movies.shape))
    Ratings = ratings['Rating'].values
    print('Ratings : {}, shape = {}'.format(Ratings,Ratings.shape))
    
    if enable_bias:
        new_Users = np.array(list(map(users_mx.get, Users)))
        new_Movies = np.array(list(map(movies_mx.get, Movies)))
    
    model = DNN(max_userid,max_movieid,DIM)
    #model = Matrix_Factorization(max_userid,max_movieid,DIM)
    model.compile(loss='mse',optimizer='adamax',metrics=[rmse]) 

    callbacks = [EarlyStopping('val_rmse',patience=2),
                 ModelCheckpoint("../model/non_normalize/model-{epoch:05d}-{rmse:.5f}-{loss:.5f}.h5",save_best_only = True)]    
    
    if enable_bias:
         history = model.fit([new_Users,new_Movies],Ratings,epochs=30,batch_size=256, validation_split=.1, verbose=1, callbacks=callbacks)
    else:
        history = model.fit([Users,Movies],Ratings,epochs=30,batch_size=256, validation_split=.1, verbose=1, callbacks=callbacks)




    ratings = normalization(ratings)    
 
    model = Matrix_Factorization(max_userid,max_movieid,DIM)
    model.compile(loss='mse',optimizer='adamax',metrics=[rmse]) 

    callbacks = [EarlyStopping('val_rmse',patience=2),
                 ModelCheckpoint("../model/normalize/model-{epoch:05d}-{rmse:.5f}-{loss:.5f}.h5",save_best_only = True)]    
    history_normalize = model.fit([Users,Movies],Ratings,epochs=30,batch_size=256, validation_split=.1, verbose=1, callbacks=callbacks)
    if plt_record:
        plt.plot(history.history['val_rmse'],label='Non-normalized RMSE')
        plt.plot(history_normalize.history['val_rmse'],label='normalized RMSE')
        plt.title('Nomalization comparison - Validaton')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend()

        plt.plot(history.history['rmse'],label='Non-normalized RMSE')
        plt.plot(history_normalize.history['rmse'],label='normalized RMSE')
        plt.title('Nomalization comparison - Train')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend()

if __name__ == '__main__':
    train_path = '../data/train.csv'
    users_path = '../data/users.csv'
    movies_path = '../data/movies.csv'

    DIM = 8
    MAX_FILE = '../model/max.csv'
    enable_bias = True
    normalize = True
    plt_record = True

    classes = ["Adventure", "Western", "Comedy", "Thriller", "Horror", "Mystery", "Crime", "Film-Noir", "Sci-Fi", "Fantasy", "Drama", "Musical", "War", "Documentary", "Children's", "Animation", "Action", "Romance"]

    main()