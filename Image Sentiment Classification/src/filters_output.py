#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
model_name = '../model/cnn_weights.002-0.70567.h5'
data_name = '../data/train.csv'
def read_data(filename, height=48, width=48):
  try:
    print('Loading X.npy & Y.npy')
    X = np.load('./data/X.npy')
    Y = np.load('./data/Y.npy')
  except:
    with open(filename, "r+") as f:
      line = f.read().strip().replace(',', ' ').split('\n')[1:]
      raw_data = ' '.join(line)
      length = width*height+1 #1 is for label
      data = np.array(raw_data.split()).astype('float').reshape(-1, length)
      X = data[:, 1:]
      Y = data[:, 0]
      # Change data into CNN format
      X = X.reshape(-1, height, width, 1)
      Y = Y.reshape(-1, 1)
      print('Saving X.npy & Y.npy')
      np.save('../data/X.npy', X) # (-1, height, width, 1)
      np.save('../data/Y.npy', Y) # (-1, 1)
  return X, Y

def main():
  emotion_classifier = load_model(model_name)
  layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)

  input_img = emotion_classifier.input
  name_ls = [name for name in layer_dict.keys() if 'leaky' in name]
  print(name_ls)
  collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

  X, Y = read_data(data_name)

  choose_id = 26000
  photo = X[choose_id].reshape(-1, height, width, 1)
  # print(photo.shape)

  for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(14, 8))
    nb_filter = min(im[0].shape[3], 32)
    for i in range(nb_filter):
      ax = fig.add_subplot(nb_filter/8, 8, i+1)
      ax.imshow(im[0][0, :, :, i], cmap='Blues')
      plt.xticks(np.array([]))
      plt.yticks(np.array([]))
      plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))

    fig.savefig(os.path.join(vis_dir,'layer{}'.format(cnt)))

if __name__ == "__main__":

  height = width = 48
  vis_dir = '../result'
  if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
  main()