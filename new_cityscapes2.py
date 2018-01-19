'''
In this script, we addopt a lot of things from : Pradyumna Reddy. We use 
cityscapes dataset that provided at: https://www.cityscapes-dataset.com/. 

In this training process, we uses both fine annotation and coarse annotation. 
Detail amount of the dataset that we use are: 
=========================================
|No | Name       | type      | amount    |
=========================================
| 1 | train      | fine      |  2.975    | 
| 2 | val        | fine      |    500    | 
| 3 | train      | coarse    |  2.975    |
| 4 | train+extra| coarse    | 19.000    |
| 5 | val        | coarse    |    500    | 
=========================================

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
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu1,floatX=float32,optimizer=fast_compile'

import time
import pylab as pl
import gc
import matplotlib.cm as cm
import itertools
import numpy as np
import matplotlib.image as mpimg
from keras import optimizers
from keras.models import model_from_json
#import theano.tensor as T
#np.random.seed(1337) # for reproducibility
import pickle
from six.moves import cPickle
from keras.layers.noise import GaussianNoise
import keras.models as models
# from keras.models import load_model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras.regularizers import ActivityRegularizer
from keras import metrics
#from keras.utils.visualize_util import plot
from numpy import array
from keras import backend as K
K.set_image_dim_ordering('tf')
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
import layers_builder as layers
import cv2
# General Parameter
# ===================
# Dataset root folder
path = '/home/adi005/Cityscapes/'

training_is = 'fine'
# Image Dimension
im_width = 713
im_height = 713 
data_shape = (im_height*im_width)
im_dimension = 3 
pretrainded = 0
nb_classes = 35
batch_size = 1
nb_epoch = 100
if training_is == 'fine' : 
    # Image Number
    train_sample = 2970
    steps = 297
    val_sample = 500
    split = train_sample/steps
    train_txt = 'train_fine_cityscapes.txt'
    val_txt = 'val_fine_cityscapes.txt'
elif training_is == 'coarse':
    train_sample = 2800
    steps = 10
    val_sample = 500
    split = train_sample/steps
    train_txt = 'train_coarse.txt'
    val_txt = 'val_coarse.txt'
else:
    train_sample = 19000
    steps = 50
    val_sample = 500
    split = train_sample/steps
    train_txt = 'train_extra.txt'
    val_txt = 'val_coarse.txt'

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
    for i in range(j,k):
        print (path + txt[i][0][1:])
        print (path + txt[i][1][1:][:-1])
        # train_data[n] = np.rollaxis(cv2.imread(path + txt[i][0][1:]),2)
        train_data[n] = cv2.imread(path+txt[i][0][1:])
        train_label[n] = binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0])
        # train_label.append(binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0]))
        print(n, ' : ' ,path + txt[i][1][1:][:-1])
        n = n + 1
    return train_data, train_label

def prep_val():
    val_data = np.zeros((val_sample, im_dimension, im_height, im_width), dtype="uint8")
    val_label = np.zeros((val_sample, im_height, im_width,nb_classes),dtype="uint8")
    # train_data = []

    #load Image from directory /Cityscapes/
    print("================================")
    print("Loading Validation Image ... \n")
    print("================================")
    with open(path+val_txt) as f:
        txt = f.readlines()
        txt = [line.split(',') for line in txt]
    for i in range(len(txt)):
        print(path + txt[1][0][1:])
        print(path + txt[i][1][1:][:-1])
        val_data[i] = np.rollaxis(cv2.imread(path + txt[i][0][1:]),2)
        #val_data[i] = cv2.imread(path+txt[i][0][1:])
        val_label [i] = binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0])
        # train_label.append(binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0]))
        print(i, ' : ' , path + txt[i][1][1:][:-1])
    return val_data, val_label

#autoencoder = segnet()
resnet_layers = 50
autoencoder = layers.build_pspnet(nb_classes=nb_classes,
                                             resnet_layers=resnet_layers,
                                             input_shape=(713, 713))
# autoencoder = segnet()
print (autoencoder.summary())
if pretrainded == 1 : 
    autoencoder.load_weights('Training_coarse_2800_samples.h5')
    print ('Wights Loaded...')

# print(autoencoder.summary())
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

time_start = time.clock()
j=0
k=0
for i in range(0,steps):
    j = i*(split)
    k = j+(split)
    txt = train_txt
    print('============================================================= \n')
    print('Training and Validation Data From = ', int(j),' : ',int(k)-1,'\n')
    print('============================================================= \n')
    # load splitted training image  
    train, train_label = prep_train(int(j),int(k),txt)
    # reshape label image to appropiate model we proposed 
    train_label = np.reshape(train_label,(int(split),data_shape,nb_classes))
    # fitting the the model
    history = autoencoder.fit(train, train_label, batch_size=batch_size, epochs=nb_epoch,
                        # uncomment this if you want use validation data
                        # verbose=1, validation_data=(val,val_label), shuffle=True)
                        # use split to validate model 
                        verbose=1, validation_split=0.2, shuffle=True)
    # history = autoencoder.fit_generator(datagen.flow(train, train_label,
    #                     batch_size=batch_size),
    #                     epochs=nb_epoch,
    #                     validation_data=(val,val_label),
    #                     workers=4)

print('\n============================================================= \n')
print('Load Testing images and Creating Validation label \n ')
print('============================================================= \n')
val, val_label = prep_val()
val_label = np.reshape(val_label,(val_sample,data_shape,nb_classes))

score = autoencoder.evaluate(val,val_label, batch_size=batch_size, show_accuracy=True, verbose=1)
print("==========================================")
print ('Loss : ',score[0])
print ('ACCS : ',score[1]*100)
print("==========================================")
time_elapsed = (time.clock() - time_start)
print('Time : ', time_elapsed,'s, or ',round(time_elapsed/3600,2),'h or , ',round(time_elapsed/3600,2)/24,'days' )
print("==========================================")

autoencoder.save_weights('Training_'+training_is+'_'+str(train_sample)+'_samples.h5')


# height = np.size(train_label[0], 0)
# width = np.size(train_label[0], 1)
# depth = np.size(train_label[0], 2)

# print(height)
# print(width)
# print(depth)
# print(train_label[1].shape)
# print(np.rollaxis(train[1],0,3).shape)

# # print(train[1])
# mpimg.imsave('coba.jpg',np.rollaxis(train[1],0,3))
