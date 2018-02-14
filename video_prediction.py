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
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time

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
batch_size = 2
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

resnet_layers = 50
model = layers.build_pspnet(nb_classes=nb_classes,
                                             resnet_layers=resnet_layers,
                                             input_shape=(713, 713))
if pretrainded == 1 : 
    model.load_weights('Final_Training_2pspnet_fine_lr_0.0025_100nb_epochadam_2970_samples.h5')
    print(model.summary())
    print ('Weights Loaded...')

unlabeled = [  0,  0,  0]               #0
ego_vehicle = [0,  0,  0]               #1
rectification_border = [0,  0,  0]      #2
out_of_roi = [0,  0,  0]                #3
static = [0,  0,  0]                    #4
dynamic = [111, 74,  0]                 #5
ground = [81,  0, 81]                   #6
road = [128, 64,128]                    #7
sidewalk = [244, 35,232]                #8
parking = [250,170,160]                 #9
rail_track = [230,150,140]              #10
building = [70, 70, 70]                 #11
wall = [102,102,156]                    #12
fence = [190,153,153]                   #13
guard_rail = [180,165,180]              #14
bridge = [150,100,100]                  #15
tunnel = [150,120, 90]                  #16
pole =[153,153,153]                     #17
polegroup = [153,153,153]               #18
traffic_light = [250,170, 30]           #19
traffic_sign = [220,220,  0]            #20
vegetation = [107,142, 35]              #21
terrain = [152,251,152]                 #22
sky = [70,130,180]                      #23
person = [220, 20, 60]                  #24 -
rider = [255,  0,  0]                   #25 -
car = [0,  0,142]                       #26 -
truck = [0,  0, 70]                     #27 - 
bus = [0, 60,100]                       #28 - 
caravan = [0,  0, 90]                   #29
trailer = [0,  0,110]                   #30
train = [0, 80,100]                     #31
motorcycle = [0,  0,230]                #32 -
bicycle = [119, 11, 32]                 #33 -
license_plate = [0,  0,142]             #34

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



cv2.namedWindow("Input")
cv2.namedWindow("SegNet")

# cap = cv2.VideoCapture(0) # Change this to your webcam ID, or file name for your video file
cap = cv2.VideoCapture('croatia.avi') # Change this to your webcam ID, or file name for your video file

rval = True

while rval:
    rval, frame = cap.read()

    if rval == False:
        break
    # print '%30s' % 'Grabbed camera frame in ', str((end - start)*1000), 'ms'
    frame = cv2.resize(frame, (713,713))
    input_image = frame
    # input_image = frame.transpose((2,0,1))
    # input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
    # input_image = np.asarray([input_image])
    input_image = np.expand_dims(input_image, axis=0)
    # print (input_image.shape)
    start = time.time()
    score=model.predict(input_image,batch_size=2)
    # print(score.shape)
    pred = visualize(np.argmax(score[0],axis=1).reshape((713,713)), False)
    end = time.time()
    seconds = end-start
    # print ("Time taken : {0} seconds".format(seconds))
    print ("video speed : ", 1/seconds," fps" )
    b,g,r = cv2.split(pred)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])
    cv2.imshow("Input", frame)
    cv2.imshow("SegNet", rgb_img)
    
    key = cv2.waitKey(1)
    if key == 27: # exit on ESC
        break
cap.release()
cv2.destroyAllWindows()