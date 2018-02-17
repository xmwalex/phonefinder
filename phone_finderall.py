#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:11:30 2018

@author: xmw
"""


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import square
from skimage.color import label2rgb

from skimage.io import imread
from sklearn.model_selection import train_test_split
import os
import sys
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, UpSampling2D
from keras.utils import np_utils

import keras.backend as K
from keras.models import Model

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD


img_rows = 320
img_cols = 480

def get_unet():
    inputs = Input((img_rows, img_cols,3))
    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)    
    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation='relu')(pool1)
    conv2 = Conv2D(64, (3, 3), padding="same", activation='relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation='relu')(pool2)
    conv3 = Conv2D(128, (3, 3), padding="same", activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation='relu')(pool3)
    conv4 = Conv2D(256, (3, 3), padding="same", activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation='relu')(pool4)
    conv5 = Conv2D(512, (3, 3), padding="same", activation='relu')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    #      Concatenate(axis=3)([residual, upconv])
    conv6 = Conv2D(256, (3, 3), padding="same", activation='relu')(up6)
    conv6 = Conv2D(256, (3, 3), padding="same", activation='relu')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), padding="same", activation='relu')(up7)
    conv7 = Conv2D(128, (3, 3), padding="same", activation='relu')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), padding="same", activation='relu')(up8)
    conv8 = Conv2D(64, (3, 3), padding="same", activation='relu')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), padding="same", activation='relu')(up9)
    conv9 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)   #9

    model = Model(inputs=inputs, outputs=conv10)
    #      `Model(inputs=/input_19, outputs=sigmoid.0)`

#    model.compile(optimizer=Adam(lr=1.0e-4), loss=dice_coef_loss, metrics=['accuracy',dice_coef])  #LUNA16
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy',dice_coef]) #ecobill

    return model


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


model = get_unet()
model.load_weights('final.h5')

input_folder=sys.argv[1]
label_path = input_folder+'/labels.txt'

with open(label_path) as f:
    content=f.readlines()

infos=[]
for line in content:
    info = line.strip('\n').split(' ')
    infos.append(info)

right=0
wrong=0
for line in infos:
#img_path = sys.argv[1]
#img_path = 'find_phone/imgs/image3.jpg'
    img_path = os.path.join(input_folder,line[0])
    bgr_img = cv2.imread(img_path)
    
    bgr_clipped = bgr_img[3:-3,5:-5]
    image=np.expand_dims(bgr_clipped,axis=0)
    #image=np.expand_dims(bgr_img,axis=0)
    pred=model.predict(image)
        #
    for i in range(0,len(pred)):
        predimg = label(pred[i,:,:,0].squeeze())
        regions = regionprops(predimg)
        if(not regions):
            print([.5,.5])
            wrong+=1
        else:
            area = [ele.area for ele in regions]
            largest_blob_ind = np.argmax(area)
            ind_pred=np.divide(np.add(list(regions[largest_blob_ind].centroid),[3,5]),[326,490])
            if(abs(float(line[1])-ind_pred[1])<0.05 and abs(float(line[2])-ind_pred[0])<0.05):
                right+=1
                print([ind_pred[1],ind_pred[0],line[1],line[2]])
            else:
                wrong+=1     
print(right)
print(wrong)
print(float(right)/(right+wrong))