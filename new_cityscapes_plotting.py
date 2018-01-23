'''
In this script, we addopt a lot of things from : Pradyumna Reddy. We use 
cityscapes dataset that provided at: https://www.cityscapes-dataset.com/. 

In this training process, we uses both fine annotation and coarse annotation. 
Detail amount of the dataset that we use are: 
===========================================================
|No | Name       | type      | amount    | Original Amount |
===========================================================
| 1 | train      | fine      |  2.975    |      2.975      |
| 2 | val        | fine      |    500    |        500      | 
| 3 | train      | coarse    |  2.975    |      2.975      | 
| 4 | train+extra| coarse    | 19.000    |     19.998      | 
| 5 | val        | coarse    |    500    |        500      |
===========================================================

Pretrained model was used to accelerate our training process. However, due to
pretrained model availability, we need to choose the most popular model. In the 
first step, we take pretrained model from model zoo, and load the matrix into 
our model, we pop the last layer to our output layer and use that model matrix
as initial value for our model. This layer really important, while we have no 
significant differences data in our experiment. In the first layer of the model,
we will have such as line, edge and several esensial feature. For example, in
Cityscapes dataset, we try to recognize human form, while in the initial model 
they have the data characteristic. 

Afterward, we use coarse dataset and its coarse annotation, to gain our model 
accuracies. The use of free trained model accelerate our learning process. So,
we dont need to train from the scratch, in other word we dont need to train 
zeros matrix. As mentioned in Table \ref{table1} we use 2975 + 19.000 coarse
data, and 500 coarse data for validation. In this training process, we use 
several optimizer such as : Adam, SGD, RMSprop. The initial learning rate $\alpha$
is 0.001 and decay 0.9. Due to our limitation resource, we use small batch size : 1. 

Our training process use full size image size : 2048 x 1080 x 3. Regarding to this
image size and resource limitation, we devide data into 5 steps for both train with
fine annotation and train with coarse annotaion, and we devide 20 steps for train extra 
with coarse annotation. In this dataset, there is no test data. So, we change validation 
data into testing data. validation with coarse annotation will be use in testing process 
with coarse annotation, and validation data with fine annotaton will be use in final testing. 
During the training process, we use 20% dataset as validation and 80% training. 
This precentage will be use in every steps in this training process. 

Our system use Ubuntu 17.04 as operating system and several main librry such as : tensorflow, 
keras, numpy etc. Our machine use 2xTitanx 1080 with 12 GB of RAM. Due to this limitation, 
we use initial epoch persteps is 200 epochs. 
'''
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cuda,floatX=float32,optimizer=fast_compile'

import time
import pylab as pl
import gc
import matplotlib.cm as cm
import itertools
import numpy as np
import matplotlib.image as mpimg
from keras import optimizers
from keras.models import model_from_json
from keras.callbacks import CSVLogger
import pickle
from six.moves import cPickle
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import metrics
from numpy import array
from keras import backend as K
K.set_image_dim_ordering('tf')
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
import layers_builder as layers
# from pspnet import PSPNet
import cv2

# General Parameter
# ===================
# Dataset root folder
path = '/home/adi005/Cityscapes/'

training_is = 'fine'
optim = 'rmsprop'
# Image Dimension
im_width = 713
im_height = 713 
data_shape = (im_height*im_width)
im_dimension = 3 
pretrainded = 1
nb_classes = 35
batch_size = 2
nb_epoch = 50

if training_is == 'fine' : 
    # Image Number
    train_sample = 2970
    steps = 10
    val_sample = 500
    split = train_sample/steps
    train_txt = 'train_fine_cityscapes.txt'
    val_txt = 'val_fine_cityscapes.txt'
elif training_is == 'coarse':
    train_sample = 2800
    nb_epoch = nb_epoch
    steps = 10
    val_sample = 500
    split = train_sample/steps
    train_txt = 'train_coarse_cityscapes.txt'
    val_txt = 'val_coarse_cityscapes.txt'
else:
    train_sample = 19000
    nb_epoch = nb_epoch - 80
    steps = 50
    val_sample = 500
    split = train_sample/steps
    train_txt = 'trainExtra_coarse_cityscapes.txt'
    val_txt = 'val_coarse_cityscapes.txt'

csv_logger = CSVLogger('log/Training_'+training_is+'_opti_'+optim+str(nb_epoch)+'-epochs_'+str(train_sample)+'_samples.log',append=True)
print("================================")
print ('Training Sample : ', train_sample)
print ('Steps           : ', steps)
print ('split per steps : ', split)
print ('train_txt       : ', train_txt)
print ('train_txt       : ', val_txt)
print ('Epoch           : ', nb_epoch)
print("================================")

def new(): 
    model = models.Sequential()
    model.add(Layer(input_shape=(im_dimension, im_height, im_width)))
    return model

def segnet():
    autoencoder = models.Sequential()
    # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
    autoencoder.add(Layer(input_shape=(im_dimension, im_height, im_width)))

    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(64, 3, 3, subsample = (1,1),border_mode='valid'))
    # print (autoencoder.summary())
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
    # print (autoencoder.summary())
    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(64, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(128, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation('relu'))
    autoencoder.add(MaxPooling2D(pool_size=(2, 2)))


    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(256, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())
    autoencoder.add(Activation('relu'))

    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(256, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())

    autoencoder.add(UpSampling2D(size=(2,2)))
    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(128, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())

    autoencoder.add(UpSampling2D(size=(2,2)))
    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(64, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())

    autoencoder.add(UpSampling2D(size=(2,2)))
    autoencoder.add(ZeroPadding2D(padding=(1,1)))
    autoencoder.add(Convolution2D(64, 3, 3, border_mode='valid'))
    autoencoder.add(BatchNormalization())
    # autoencoder.add(Layer(input_shape=(im_dimension, im_height,im_width)))
    autoencoder.add(Convolution2D(nb_classes, 1, 1, border_mode='valid',))
    #import ipdb; ipdb.set_trace()
    autoencoder.add(Reshape((nb_classes,data_shape), input_shape=(nb_classes,im_height,im_width)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    return autoencoder



def binarylab(labels):
    x = np.zeros([im_height,im_width,nb_classes],dtype="uint8")
    for i in range(im_height):
        for j in range(im_width):
            if labels[i][j]>35:
                labels[i][j] = 0
                x[i,j,labels[i][j]]=1
                print ('More than 35')
            else:
                x[i,j,labels[i][j]]=1
    return x

def prep_train(j,k,train_txt):
    train_data = np.zeros((int(split), im_height, im_width, im_dimension), dtype="uint8")
    train_label = np.zeros((int(split), im_height, im_width,nb_classes),dtype="uint8")
    # train_data = []

    #load Image from directory /Cityscapes/
    print("================================")
    print("Loading Training Image ... \n")
    print("================================")
    with open(path+train_txt) as f:
        txt = f.readlines()
        # print(txt,'\n')
        txt = [line.split(',') for line in txt]
    n = 0
    print('Loading Training Images from',j, ' to ',k)
    for i in range(j,k):
        # print (path + txt[i][0][1:])
        # print (path + txt[i][1][1:][:-1])
        # train_data[n] = np.rollaxis(cv2.imread(path + txt[i][0][1:]),2)
        train_data[n] = cv2.imread(path+txt[i][0][1:])
        train_label[n] = binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0])
        # train_label.append(binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0]))
        # print(n, ' : ' ,path + txt[i][1][1:][:-1])
        n = n + 1
        s = str(i) + '/'+str(k)                       # string for output
        print('{0}\r'.format(s), end='')        # just print and flush
        time.sleep(0.2)
    print('\nLoaded Training Image from',j, ' to ', k)
    return train_data, train_label

def prep_val(j,k,train_txt, splitval):
    # val_data = np.zeros((val_sample, im_dimension, im_height, im_width), dtype="uint8")
    # val_label = np.zeros((val_sample, im_height, im_width,nb_classes),dtype="uint8")
    val_data = np.zeros((int(splitval), im_height, im_width, im_dimension), dtype="uint8")
    val_label = np.zeros((int(splitval), im_height, im_width,nb_classes),dtype="uint8")
    # train_data = []

    #load Image from directory /Cityscapes/
    print("================================")
    print("Loading Testing Image ... \n")
    print("================================")
    with open(path+val_txt) as f:
        txt = f.readlines()
        txt = [line.split(',') for line in txt]
    print('Loading Testing Images from',j, ' to ',k)
    n = 0
    for i in range(j,k):
        # print(path + txt[1][0][1:])
        # print(path + txt[i][1][1:][:-1])
        # val_data[n] = np.rollaxis(cv2.imread(path + txt[i][0][1:]),2)
        val_data[n] = cv2.imread(path+txt[i][0][1:])
        val_label [n] = binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0])
        # train_label.append(binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0]))
        n = n + 1
        # print(i, ' : ' , path + txt[i][1][1:][:-1])
        s = str(i) + '/'+str(k)                       # string for output
        print('{0}\r'.format(s), end='')        # just print and flush
        time.sleep(0.2)
    print('\nLoaded Testing Images from',j, ' to ',k)
    return val_data, val_label

#autoencoder = segnet()
resnet_layers = 50
model = layers.build_pspnet(nb_classes=nb_classes,
                                             resnet_layers=resnet_layers,
                                             input_shape=(713, 713))
# # autoencoder = segnet()
# # print (autoencoder.summary())
if pretrainded == 1 : 
    model.load_weights('Training_fine_rmsprop_2970_samples_8_splits.h5')
    print ('Weights Loaded...')

unlabeled = [  0,  0,  0]
ego_vehicle = [0,  0,  0]
rectification_border = [0,  0,  0]
out_of_roi = [0,  0,  0]
static = [0,  0,  0]
dynamic = [111, 74,  0]
ground = [81,  0, 81]
road = [128, 64,128]
sidewalk = [244, 35,232]
parking = [250,170,160]
rail_track = [230,150,140]
building = [70, 70, 70]
wall = [102,102,156]
fence = [190,153,153]
guard_rail = [180,165,180]
bridge = [150,100,100]
tunnel = [150,120, 90]
pole =[153,153,153]
polegroup = [153,153,153]
traffic_light = [250,170, 30]
traffic_sign = [220,220,  0]
vegetation = [107,142, 35]
terrain = [152,251,152]
sky = [70,130,180]
person = [220, 20, 60]
rider = [255,  0,  0]
car = [0,  0,142]
truck = [0,  0, 70]
bus = [0, 60,100]
caravan = [0,  0, 90]
trailer = [0,  0,110]
train = [0, 80,100]
motorcycle = [0,  0,230]
bicycle = [119, 11, 32]
license_plate = [0,  0,142]

label_colours = np.array([unlabeled , ego_vehicle ,rectification_border ,out_of_roi ,static ,dynamic ,ground ,road ,sidewalk ,parking ,rail_track ,building ,wall ,fence , guard_rail ,bridge ,tunnel ,pole ,polegroup ,traffic_light ,traffic_sign ,
                        vegetation ,terrain ,sky ,person , rider , car , truck , bus , caravan , trailer , train , motorcycle , bicycle ,license_plate])


def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,34):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((713, 713, 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

print (model.summary())
txt = train_txt
train, train_label = prep_train(0,2,txt)

print ('==================\n')
print ('Predict Image ...')
print ('==================\n')
coba=model.predict(train,batch_size=batch_size)
print ('==================\n')
print ('Visualize Image ...')
print ('==================\n')
pred = visualize(np.argmax(coba[1],axis=1).reshape((713,713)), False)

fig = plt.figure()
plt.subplot(121)
plt.imshow(train[1])
plt.subplot(122)
plt.imshow(pred)
plt.show()


# print (pred.shape())
# plt.imshow(pred)
# plt.imshow(cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
# plt.show()
