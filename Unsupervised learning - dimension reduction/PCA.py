import skimage.io
from skimage import transform
import numpy as np
import sys
import os
from os import listdir
img_folder = "../data/PCA"
ids = [23, 96, 187, 253]

image_names = listdir(img_folder)
# load all fimport skimage.io
from skimage import transform
import numpy as np
import sys
import os
from os import listdir
img_folder = "../data/PCA"
ids = [23, 96, 187, 253]

image_names = listdir(img_folder)
# load all faces array
image_X = []
def img_resize(img):
    img = transform.resize(img,(300,300,3))
    return img
def image_clip(x):
    x -= np.min(x)
    x /= np.max(x)
    x = (x * 255).astype(np.uint8)
    x = np.reshape(x, (300, 300, 3))
    return x

for name in image_names:
    single_img = skimage.io.imread(os.path.join(img_folder,name))
    single_img = img_resize(single_img)
    image_X.append(single_img)
    
image_flat = np.reshape(image_X,(415,-1))
mean_face = np.mean(image_flat,axis=0)
image_center = image_flat - mean_face

print("Running SVD........")
# u : Unitary matrix eigenvectors in columns
# d : list of the singulat values,sorted in descendinng order
U, S, V = np.linalg.svd(image_center.T, full_matrices=False)

print("U shape",U.shape)
print("S shape",S.shape)
print("V shape",V.shape)

# mean
skimage.io.imsave("../result/mean.jpg", np.reshape(mean_face,(300,300,3)))

#Eigen_vector
for i in range(4):
    X = image_clip(U[:,i])
    skimage.io.imsave("../result/eigen_vectoy{}.jpg".format(i), np.reshape(X,(300,300,3)))
#skimage.io.imsave("../result/eigen_vector.jpg", np.reshape(mean_face,(300,300,3))

# reconstruct 
top = 4

for id in ids:
    input_img = skimage.io.imread(os.path.join(img_folder,'{}.jpg').format(id))
    input_img = img_resize(input_img)
    input_img = input_img.flatten()
    input_img_center = input_img - mean_face
    weights = np.dot(input_img_center, U[:, :top])
    recon = mean_face + np.dot(weights, U[:, :top].T)
    recon = image_clip(recon)
    skimage.io.imsave("../result/reconstruction_{}.jpg".format(id), recon)
aces array
image_X = []
def img_resize(img):
    img = transform.resize(img,(300,300,3))
    return img
def image_clip(x):
    x -= np.min(x)
    x /= np.max(x)
    x = (x * 255).astype(np.uint8)
    x = np.reshape(x, (300, 300, 3))
    return x

for name in image_names:
    single_img = skimage.io.imread(os.path.join(img_folder,name))
    single_img = img_resize(single_img)
    image_X.append(single_img)
    
image_flat = np.reshape(image_X,(415,-1))
mean_face = np.mean(image_flat,axis=0)
image_center = image_flat - mean_face

print("Running SVD........")
# u : Unitary matrix eigenvectors in columns
# d : list of the singulat values,sorted in descendinng order
U, S, V = np.linalg.svd(image_center.T, full_matrices=False)

print("U shape",U.shape)
print("S shape",S.shape)
print("V shape",V.shape)

# mean
skimage.io.imsave("../result/mean.jpg", np.reshape(mean_face,(300,300,3)))

#Eigen_vector
for i in range(4):
    X = image_clip(U[:,i])
    skimage.io.imsave("../result/eigen_vectoy{}.jpg".format(i), np.reshape(X,(300,300,3)))
#skimage.io.imsave("../result/eigen_vector.jpg", np.reshape(mean_face,(300,300,3))

# reconstruct 
top = 4

for id in ids:
    input_img = skimage.io.imread(os.path.join(img_folder,'{}.jpg').format(id))
    input_img = img_resize(input_img)
    input_img = input_img.flatten()
    input_img_center = input_img - mean_face
    weights = np.dot(input_img_center, U[:, :top])
    recon = mean_face + np.dot(weights, U[:, :top].T)
    recon = image_clip(recon)
    skimage.io.imsave("../result/reconstruction_{}.jpg".format(id), recon)
