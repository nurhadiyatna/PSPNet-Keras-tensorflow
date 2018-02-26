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
#import theano.tensor as T
#np.random.seed(1337) # for reproducibility
from keras.callbacks import CSVLogger
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
dataset = 'camvid'
model_names = 'pspnet'
optim = 'adam'
im_width = 713
im_height = 713
data_shape = (im_height*im_width)
im_dimension = 3 
pretrainded = 1
nb_classes = 2
batch_size = 2
nb_epoch = 100
learning_rate = 0.001
if dataset == 'camvid':
    PATH_Camvid = '/home/adi005/CamVid/'
    train_val_txt = 'camvid_train_val_test.txt'
    train_sample = 701
    steps = 1
    # val_sample = 500
    split = train_sample/steps
elif dataset == 'voc':
    PATH_VOC = '/home/adi005/VOC2010/'
    train_val_txt = 'trainval.txt'
    train_sample = 1928
    steps = 2
    # val_sample = 500
    split = train_sample/steps


csv_logger = CSVLogger('log_voc_camvid/Training_2_Class'+model_names+'_'+dataset+'_'+optim+'_'+str(nb_epoch)+'_epochs_'+str(train_sample)+'_samples.log',append=True)
print ("================================")
print ('Model Name      : ', model_names)
print ('Training Sample : ', train_sample)
print ('Steps           : ', steps)
print ('split per steps : ', split)
print ('train_txt       : ', train_val_txt)
print ('Epoch           : ', nb_epoch)
print ("================================")


def segnet():
    autoencoder = models.Sequential()
    # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
    autoencoder.add(Layer(input_shape=(im_height, im_width,im_dimension)))

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

def excludingClassVOC(labels):
    labels[(labels[:,:,0] !=128)|(labels[:,:,1]!=128)|(labels[:,:,2]!=192)]=0
    labels[(labels[:,:,0] ==128)&(labels[:,:,1]==128)&(labels[:,:,2]==192)]=1
    return labels

def excludingClassCamvid(labels):
    labels[(labels[:,:,0] !=9)|(labels[:,:,1]!=9)|(labels[:,:,2]!=9)]=0
    labels[(labels[:,:,0] ==9)&(labels[:,:,1]==9)&(labels[:,:,2]==9)]=1
    return labels

def binarylab(labels):
    x = np.zeros([im_height,im_width,nb_classes],dtype="uint8")
    # ignore = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,29,30,31,33,34]
    # labels[labels>34]=0
    for i in range(im_height):
        for j in range(im_width):
        #     if labels[i][j]>34:
        #         labels[i][j] = 0
        #         x[i,j,labels[i][j]]=1
        #     else:
            x[i,j,labels[i][j]]=1
    return x

def prep_train_voc(j,k,train_txt):
    train_data = np.zeros((int(split), im_height, im_width, im_dimension), dtype="uint8")
    train_label = np.zeros((int(split), im_height, im_width,nb_classes),dtype="uint8")
    # train_data = []

    #load Image from directory /Cityscapes/
    print("================================")
    print("Loading voc2010 Image ... \n")
    print("================================")
    with open(PATH_VOC+train_txt) as f:
        txt = f.readlines()
        txt = [line.split(',') for line in txt]
    print('Loading Testing Images from',j, ' to ',k)
    # print(txt,'\n')
    n = 0
    for i in range(j,k):
        train_data[n] = cv2.resize(cv2.imread(PATH_VOC+txt[i][0][0:]+'.jpg'),(713,713))
        train_label [n] = binarylab(excludingClassVOC(cv2.resize(cv2.imread(PATH_VOC + txt[i][1][0:][:-1]),(713,713)))[:,:,0])
        n = n + 1
        s = str(i) + '/'+str(k)                       # string for output
        print('{0}\r'.format(s), end='')        # just print and flush
        time.sleep(0.2)
    print('Loaded Testing Images from',j, ' to ',k)
    return train_data, train_label

def prep_train_camvid(j,k,train_txt):
    train_data = np.zeros((int(split), im_height, im_width, im_dimension), dtype="uint8")
    train_label = np.zeros((int(split), im_height, im_width,nb_classes),dtype="uint8")
    # train_data = []

    #load Image from directory /Cityscapes/
    print("================================")
    print("Loading CamVid Image ... \n")
    print("================================")
    with open(PATH_Camvid+train_txt) as f:
        txt = f.readlines()
        txt = [line.split(',') for line in txt]
    print('Loading CamVid Images from',j, ' to ',k)
    # print(txt,'\n')
    n = 0
    for i in range(j,k):
        train_data[n] = cv2.resize(cv2.imread(PATH_Camvid+txt[i][0][0:]),(713,713))
        train_label [n] = binarylab(excludingClassCamvid(cv2.resize(cv2.imread(PATH_Camvid + txt[i][1][0:][:-1]),(713,713)))[:,:,0])
        n = n + 1
        s = str(i) + '/'+str(k)                       # string for output
        print('{0}\r'.format(s), end='')        # just print and flush
        time.sleep(0.2)
    print('Loaded Camvid Images from',j, ' to ',k)
    return train_data, train_label


if model_names =='segnet':
    model = segnet()
else:
    resnet_layers = 50
    model = layers.build_pspnet(nb_classes=nb_classes,
                                                 resnet_layers=resnet_layers,
                                                 input_shape=(713, 713))
# # Save model into Json
# model_json = model.to_json()
# with open("PSPNet_Resnet50.json", "w") as json_file:
#     json_file.write(model_json)
#     print ("Model Save to Json")
# autoencoder = segnet()
# print (autoencoder.summary())
if pretrainded == 1 : 
    model.load_weights('Final_Training_2_Class_pspnet_coarse_lr_0.001_50nb_epochadam_2800_samples.h5')
    print ('Weights Loaded...')

# print(autoencoder.summary())

#create optimizer
if optim == 'sgd' : 
    optimi = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
elif optim == 'rmsprop':
    optimi = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
elif optim == 'adam':
    optimi = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
elif optim == 'adadelta':
    optimi = optimizers.Adadelta(lr=learning_rate, rho=0.95, epsilon=None, decay=0.0)
elif optim == 'adagrad':
    optimi = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
else:
    optimi = optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

#comipile generated model
model.compile(loss="categorical_crossentropy", optimizer=optimi, metrics=['accuracy'])

time_start = time.clock()
j=0
k=0
for i in range(0,steps):
    j = i*(split)
    k = j+(split)
    txt = train_val_txt
    print('============================================================= \n')
    print('Training and Validation Data From = ', int(j),' : ',int(k)-1,'\n')
    print('============================================================= \n')
    # load splitted training image  
    if dataset == 'camvid': 
        train, train_label = prep_train_camvid(int(j),int(k),txt)
    elif dataset == 'voc':
        train, train_label = prep_train_voc (int(j),int(k),txt)
    # reshape label image to appropiate model we proposed 
    train_label = np.reshape(train_label,(int(split),data_shape,nb_classes))
    # fitting the the model
    history = model.fit(train, train_label, batch_size=batch_size, epochs=nb_epoch,callbacks = [csv_logger],
                        # uncomment this if you want use validation data
                        # verbose=1, validation_data=(val,val_label), shuffle=True)
                        # use split to validate model 
                        verbose=1, validation_split=0.2, shuffle=True)
    model.save_weights('models/Training_2_Class_'+model_names+'_'+Dataset+'_'+optim+'_'+str(train_sample)+'_samples_'+str(i)+'_splits.h5')
    print ('This is ',i,'of splits, and',split-i,'remaining')
    del train
    del train_label
    # history = autoencoder.fit_generator(datagen.flow(train, train_label,
    #                     batch_size=batch_size),
    #                     epochs=nb_epoch,
    #                     validation_data=(val,val_label),
    #                     workers=4)

model.save_weights('models/final/Final_Training_2_Class_'+model_names+'_'+Dataset+'_'+'lr_'+str(learning_rate)+'_'+str(nb_epoch)+'nb_epoch'+optim+'_'+str(train_sample)+'_samples.h5')
# j = 0
# k = 0
# splitval = val_sample/steps
# for i in range(0,steps):
#     j = i*(splitval)
#     k = j+(splitval)
#     txt = val_txt
#     print('\n============================================================= \n')
#     print('Load Testing images and Creating Validation label \n ')
#     print('============================================================= \n')
#     val, val_label = prep_val(int(j),int(k),txt,int(splitval))
#     val_label = np.reshape(val_label,(int(splitval),data_shape,nb_classes))
#     score = model.evaluate(val,val_label, batch_size=batch_size, verbose=1)
#     # del val_label
    # del val
    # print("==========================================")
    # print ('Loss : ',score[0])
    # print ('ACCS : ',score[1]*100)
    # print("==========================================")

# time_elapsed = (time.clock() - time_start)
# print('Time : ', time_elapsed,'s, or ',round(time_elapsed/3600,2),'h or , ',round(time_elapsed/3600,2)/24,'days' )
# print("==========================================")

#====================================
#plotting result of training process
#====================================

# data=np.loadtxt('Training_pspnet_fine_100_epochs_2970_samples.log',skiprows=1,delimiter=',')

# Training=data[:,1]
# Validation=data[:,3]

# x = np.arange(len(Training))

# plt.plot(x, np.sort(Validation))
# # plt.plot(x, Validation)
# # plt.plot(x, Training)
# plt.plot(x, np.sort(Training))
# plt.show()

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
