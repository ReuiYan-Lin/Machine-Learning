# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 13:41:11 2018

@author: pc
"""

import sys, os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
path = "../data/test.csv"
width = int(48)
height = int(48)
num_classes = 7
model_name = "../model/cnn_weights.002-0.70567.h5"
file_path = "../result/cnn_reslut.csv"
def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
      os.makedirs(directory)

def main():
  # Read the test data
  with open(path, "r+") as f:
      line = f.read().strip().replace(',', ' ').split('\n')[1:]
      raw_data = ' '.join(line)
      length = width*height+1 #1 is for label
      data = np.array(raw_data.split()).astype('float').reshape(-1, length)
      X = data[:, 1:]
      X /= 255     
  model = load_model(model_name)
  
  X = X.reshape(X.shape[0],height,width,1)
  ans = model.predict_classes(X)
  ans = list(ans)
  
  ensure_dir(file_path )
  
  result = []
  
  for index,value in enumerate(ans):
      result.append("{0},{1}".format(index, value))
  with open(file_path,"w+") as f:
      f.write("id,label\n")
      f.write("\n".join(result))
if __name__ == '__main__':
    main()