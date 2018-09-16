# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 23:13:01 2018

@author: acer
"""

import sys,os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np 

model_name = './ckpt/cnn_weights.002-0.70567.h5'
nb_filter = 32
RECORD_FREQ = 10
NUM_STEPS = 100
num_step = 100
adversial_dir = './adversial_image'
data_name = './data/train.csv'
one_img = './data/one_img.csv'
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
      np.save('./data/X.npy', X) # (-1, height, width, 1)
      np.save('./data/Y.npy', Y) # (-1, 1)
  return X, Y
def read_one_img(img,height=48,width=48):
    with open(img,'r+') as f:
        raw_data = ' '+f.read().strip().replace(',',' ')
        length = height*width+1
        data = np.array(raw_data.split()).astype('float').reshape(-1, length)
        X = data[:,1:]
        Y = data[:,0]
        X = X.reshape(-1,height,width,1)
        Y = Y.reshape(-1,1)
    return X,Y


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x

def normalize(x):
  # utility function to normalize a tensor by its L2 norm
  return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
  """
  Implement this function!
  """
  step = 1e-2
  for i in range(num_step):
    loss_value, grads_value = iter_func([input_image_data, 0])
    input_image_data += grads_value * step

    filter_images = (input_image_data, loss_value)
    print('#{}, loss rate: {}'.format(i, loss_value))
  return filter_images

def main():
  emotion_classifier = load_model(model_name)
  layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
  input_img = emotion_classifier.input

  #name_ls = [name for name in layer_dict.keys() if 'leaky' in name]
  collect_layers = [ emotion_classifier.output ]
  print(collect_layers)
  for cnt, c in enumerate(collect_layers):
    filter_imgs = []
    for class_idx in range(classes):
      input_img_data = np.expand_dims(X[picked_img_id], axis=0) # picked image
      target = K.mean(c[:, class_idx])
      grads = normalize(K.gradients(target, input_img)[0])
      iterate = K.function([input_img, K.learning_phase()], [target, grads])

      ###
      "You need to implement it."
      print('===class:{}==='.format(class_idx))
      filter_imgs.append(grad_ascent(num_step, input_img_data, iterate))
      ###
    print('Finish gradient')

    for i, emotion in enumerate(class_names):
      print('In the class #{}'.format(class_names[i]))
      fig = plt.figure(figsize=(8, 8))
      raw_img = filter_imgs[i][0].squeeze()
      plt.imshow(deprocess_image(raw_img), cmap='gray')
      plt.xticks(np.array([]))
      plt.yticks(np.array([]))
      plt.xlabel('{}'.format(emotion))
      plt.tight_layout()
      fig.savefig(os.path.join(adversial_dir,'_{}_'.format(i)))
      print('adversial_path : %s'%os.path.join(adversial_dir,'_{}_'.format(i)))
if __name__ == '__main__':
    
    class_names = ['Angry']
    num_step = 50
    classes = len(class_names)
    print('class : %d'%classes)
    '''
    X,Y = read_data(data_name)
    
    base_id = -3000
    picked_img_id = 13+base_id
    '''
    X,Y = read_one_img(one_img)
    picked_img_id = 0
    fig = plt.figure(figsize = (8,8))
    plt.imshow(X[picked_img_id].squeeze(),cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    fig.savefig(os.path.join(adversial_dir,'_original_'))
    print('ori_path : %s'%os.path.join(adversial_dir,'_original_'))
    
    X /= 255
    
    main()
    