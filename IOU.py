import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from keras import backend as K
K.set_image_dim_ordering('tf')

im_width = 713
im_height = 713 
nb_classes = 35
path = '/home/gandalf/hasil_training/histogram/'
val_txt = 'val_fine_cityscapes.txt'
steps = 500
val_sample = 500

def binarylab(labels):
    x = np.zeros([im_height,im_width,nb_classes],dtype="uint8")
    labels[labels>34]=0
    for i in range(im_height):
        for j in range(im_width):
            # if labels[i][j]>35:
            #     labels[i][j] = 0
            #     x[i,j,labels[i][j]]=1
            #     print ('More than 35')
            # else:
            x[i,j,labels[i][j]]=1
    return x

def prep_val(j,k,train_txt, splitval):

    val_data = np.zeros((int(splitval), im_height, im_width, 3), dtype="uint8")
    val_label = np.zeros((int(splitval), im_height, im_width,3),dtype="uint8")
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
        val_data[n] = cv2.imread(path+txt[i][0][1:])
        val_label [n] = cv2.imread(path + txt[i][1][1:][:-1])
    print('Loaded Testing Images from',j, ' to ',k)
    return val_data, val_label

j = 0
k = 0
l = 0
splitval = val_sample/steps
conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
IOU_mean = np.zeros((nb_classes), dtype = float)
for i in range(0,steps):
    j = i*(splitval)
    k = j+(splitval)
    txt = val_txt
    print('\n============================================================= \n')
    print('Load Testing images and Creating Validation label \n ')
    print('============================================================= \n')
    val, val_label = prep_val(int(j),int(k),txt,int(splitval))
    val_predict = cv2.imread('result/'+str(i)+'.png')
    flat_label = np.ravel(val_label[0][:,:,0])
    flat_pred = np.ravel(val_predict[:,:,0])
    for p, l in zip(flat_pred, flat_label):
        if l == 255:
            continue
        if l < nb_classes and p < nb_classes:
            conf_m[l, p] += 1
        else:
            print('Invalid entry encountered, skipping! Label: ')
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U

    meanIOU = np.mean(IOU)
    # IOU_mean = 0
    for i in range (len(IOU)):
        if np.isfinite(IOU[i]):
        #     IOU[i] = 1
            print ('IoU object : ', i, ' : ', (IOU[i]*100), '%')
        # if i>0 : 
            IOU_mean[i] = IOU_mean[i-1]+IOU[i]
            # print (IOU_mean)
    # plt.imshow(val_predict)
    # plt.show()
    # cv2.imwrite('result/'+str(l)+'.png',val_label[0])
    l = l + 1

for i in range (nb_classes):
    print ('Mean IOU [',i,']',(IOU_mean[i]/500)*100)
