from __future__ import absolute_import
from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'

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


W = 2048
H = 1024
image_dim = 713
shows = 0
w_tail = W-(image_dim*2)
ketiga = (image_dim*2)-(image_dim-w_tail)
h_tail = H-image_dim

training_is = 'fine'
model_names = 'pspnet'
optim = 'sgd'
# Image Dimension
if model_names == 'segnet':
    im_width = 512
    im_height = 256
    path = '/home/adi005/Cityscapes_segnet/' 
elif model_names == 'pspnet':
    im_width = 713
    im_height = 713 
    path = '/home/adi005/Cityscapes713/' 
else:
    im_width = 433
    im_height = 433 

data_shape = (im_height*im_width)
im_dimension = 3 
pretrainded = 1
nb_classes = 35
batch_size = 1
nb_epoch = 100
#create log in csv style

if training_is == 'fine' : 
    # Image Number
    train_sample = 2970
    steps = 50
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


def crop(input):
	cropped1 = input[0:713, 0:713]
	cropped2 = input[0:713, 713:713*2]
	cropped3 = input[0:713, ketiga:2048]
	cropped4 = input[311:1024, 0:713]
	cropped5 = input[311:1024, 713:713*2]
	cropped6 = input[311:1024, ketiga:2048]
	if shows==1: 
		cv2.imshow("cropped1", cropped1)
		cv2.imwrite('cropped1.png',cropped1)
		# cv2.imshow('image',input)
		cv2.waitKey(0)
		cv2.imshow("cropped2", cropped2)
		cv2.imwrite('cropped2.png',cropped2)
		# cv2.imshow('image',input)
		cv2.waitKey(0)
		cv2.imshow("cropped3", cropped3)
		cv2.imwrite('cropped3.png',cropped3)
		# cv2.imshow('image',input)
		cv2.waitKey(0)
		cv2.imshow("cropped4", cropped4)
		cv2.imwrite('cropped4.png',cropped4)
		# cv2.imshow('image',input)
		cv2.waitKey(0)
		cv2.imshow("cropped5", cropped5)
		cv2.imwrite('cropped5.png',cropped5)
		# cv2.imshow('image',input)
		cv2.waitKey(0)
		cv2.imshow("cropped6", cropped6)
		cv2.imwrite('cropped6.png',cropped6)
		# cv2.imshow('image',input)
		cv2.waitKey(0)
		cv2.imshow('Original image',input)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	return cropped1,cropped2,cropped3,cropped4,cropped5,cropped6

def merge(input1,input2,input3,input4,input5,input6):
	temp = np.zeros((1024, 2048, 3), dtype="float64")
	temp [0:713, 0:713] 		= input1
	temp [0:713, 713:713*2]		= input2
	temp [0:713, ketiga:2048]	= input3
	temp [311:1024, 0:713]		= input4
	temp [311:1024, 713:713*2]	= input5
	temp [311:1024, ketiga:2048]= input6
	return temp

resnet_layers = 50
model = layers.build_pspnet(nb_classes=nb_classes,
                                             resnet_layers=resnet_layers,
                                             input_shape=(713, 713))

if pretrainded == 1 : 
    model.load_weights('Final_Training_pspnet_fine_100nb_epochrmsprop_2970_samples.h5')
    print ('Weights Loaded...')
    # sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])

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

def visualize_id(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    # for l in range(0,34):
    #     r[temp==l]=label_colours[l,0]
    #     g[temp==l]=label_colours[l,1]
    #     b[temp==l]=label_colours[l,2]

    rgb = np.zeros((713, 713, 3))
    rgb[:,:,0] = (r)#[:,:,0]
    rgb[:,:,1] = (g)#[:,:,1]
    rgb[:,:,2] = (b)#[:,:,2]
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def prep_val(j,k,train_txt, splitval):
    # val_data = np.zeros((val_sample, im_dimension, im_height, im_width), dtype="uint8")
    # val_label = np.zeros((val_sample, im_height, im_width,nb_classes),dtype="uint8")
    val_data = np.zeros((int(splitval), im_height, im_width, im_dimension), dtype="uint8")

    # train_data = []

    #load Image from directory /Cityscapes/
    print("================================")
    print("Loading Validation Image ... \n")
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
        # val_label [n] = binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0])
        # train_label.append(binarylab(cv2.imread(path + txt[i][1][1:][:-1])[:,:,0]))
        n = n + 1
        s = str(i) + '/'+str(k)                       # string for output
        print('{0}\r'.format(s), end='')        # just print and flush
        time.sleep(0.2)
        # print(i, ' : ' , path + txt[i][1][1:][:-1])
    print('Loaded Testing Images from',j, ' to ',k)
    return val_data


j = 0
k = 0
splitval = val_sample/steps
for i in range(0,steps):
    j = i*(splitval)
    k = j+(splitval)
    txt = val_txt
    print('\n============================================================= \n')
    print('Load Testing images and Creating Validation label \n ')
    print('============================================================= \n')
    val= prep_val(int(j),int(k),txt,int(splitval))
    # print(file_name[i])
    score=model.predict(val,batch_size=batch_size)
    seg = np.zeros((int(splitval),713, 713, 3), dtype="float64")
    seg2 = np.zeros((int(splitval),713, 713, 3), dtype="uint8")
    for m in range(0,int(splitval)):
        id_pred = visualize_id(np.argmax(score[m],axis=1).reshape((713,713)),False)
        pred = visualize(np.argmax(score[m],axis=1).reshape((713,713)), False)
        seg2[m] = id_pred
        seg [m] = pred
        if m == 9:
            fig = plt.figure()
            plt.subplot(131)
            plt.imshow(val[1])
            plt.subplot(132)
            plt.imshow(seg[1])
            plt.subplot(133)
            plt.imshow(seg2[1])
            # plt.imshow(seg[5])
            plt.show()
        # plt.imsave(str(splitval+m)+'.png')
        # Next step is to write whole tested image into a same name as the id

'''
This is sliced prediction block 
'''

# input = cv2.imread('weimar_000139_000019_leftImg8bit.png')        
# input_sliced = np.zeros((6,713, 713, 3), dtype="uint8")
# input_sliced[0],input_sliced[1], input_sliced[2],input_sliced[3],input_sliced[4],input_sliced[5] = crop(input)

# print ('==================\n')
# print ('Predict Image ...')
# print ('==================\n')
# score=model.predict(input_sliced,batch_size=batch_size)
# print ('==================\n')
# print ('Visualize Image ...')
# print ('==================\n')
# # print (coba)
# seg = np.zeros((6,713, 713, 3), dtype="float64")
# seg2 = np.zeros((6,713, 713, 3), dtype="uint8")
# for i in range(0,6):
# 	id_pred = visualize_id(np.argmax(score[i],axis=1).reshape((713,713)),False)
# 	pred = visualize(np.argmax(score[i],axis=1).reshape((713,713)), False)
# 	seg2[i] = id_pred
# 	seg [i] = pred

# # cv2.imshow('hasil akhir',seg[3])
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# hasil = merge(seg[0],seg[1], seg[2],seg[3],seg[4],seg[5])
# plt.imshow(seg2[1])
# # plt.imshow(seg[5])
# plt.show()
# # cv2.imshow('Segmentation',hasil)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imsave('FUll_merged.png', hasil)
