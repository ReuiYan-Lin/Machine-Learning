# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 22:39:20 2018

@author: acer
"""

import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model,load_model
from keras import backend as K

test_path = './data/test.csv'
users_path = './data/users.csv'
movies_path = './data/movies.csv'
model_path = 'model/test/model-00032-0.84966-0.75923.h5'
max_userid = 6040
max_movieid = 3952

def predict_rating(trained_model,userid,movieid):
    return rate(trained_model,userid -1,movieid -1)

def rate(model,user_id,item_id):
    return model.predict([np.array([user_id]),np.array([item_id])])[0][0]

def rmse(y_true,y_pred):
    y_pred = K.clip(y_pred,1.,5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def main():
    test_data = pd.read_csv(test_path,usecols=['UserID','MovieID'])
    print('{} testing data load.'.format(test_data.shape[0]))
    
    users = pd.read_csv(users_path,sep='::',engine='python',usecols = ['UserID','Gender','Age','Occupation','Zip-code'])
    print('{} description of {} users loaded.'.format(len(users),max_userid))

    movies = pd.read_csv(movies_path,sep='::',engine='python',usecols = ['movieID','Title','Genres'])
    print('{} description of {} movies loaded'.format(len(movies),max_movieid))
    
    model = load_model(model_path,custom_objects={'rmse':rmse})
    print('Loading model weights...')
    
    recommendations = pd.read_csv(test_path,usecols=['TestDataID'])
    recommendations['Rating'] = round(test_data.apply(lambda x:predict_rating(model,x['UserID'],x['MovieID']),axis=1),1)
    recommendations.to_csv('./result/ans.csv',index=False,columns=['TestDataID','Rating'])

if __name__ == '__main__':
    main()