

from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense,Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import keras.initializers
from keras.optimizers import Adam
import pickle
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import scipy.io as sio 
import cv2
import scipy.linalg as linalg
import time


def get_var(name,all_vars):
        for i in range(len(all_vars)):
            if all_vars[i].name.startswith(name):
                return all_vars[i]



def dnn_block(cnn_input):
    conv1= Convolution2D(32, (5, 5), padding='same', kernel_initializer='glorot_normal',bias_initializer='zeros')(cnn_input)
    relu1=Activation('relu')(conv1)
    conv2= Convolution2D(16, (5, 5), padding='same', kernel_initializer='glorot_normal',bias_initializer='zeros')(relu1)
    relu2=Activation('relu')(conv2)
    conv3= Convolution2D(1, (5, 5), padding='same', kernel_initializer='glorot_normal',bias_initializer='zeros')(relu2)
    relu3=Activation('relu')(conv3)
    
    return relu3


def bulid_reconstruction(X_in,PHI_ex,M,n):
    x_in=Reshape([1024,])(X_in)
    l = Dense(M,kernel_initializer=keras.initializers.Constant(value=PHI_ex))
    y=l(x_in)
    p=l.get_weights()
    a=np.array(p[0])
    x_hat=Dense(1024,activation='relu')(y)
    X_hat=Reshape([32,32,1])(x_hat)
    layers = []
    layers.append(X_hat)
    for i  in range(n):
        relu3=dnn_block(layers[-1])
        layers.append(relu3)
    return layers,a




