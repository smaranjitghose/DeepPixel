import numpy
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape, Activation, Dropout, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.clear_session()
def InstantiateModel(inputs):
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(inputs)
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv2D(512, (3,3), activation = 'relu', padding = 'same')(conv5)

    up6 = Conv2D(256, (2,2), activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(256,(3, 3), activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(conv6)

    up7 = Conv2D(128, (2,2), activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv7)

    up8 = Conv2D(64, (2,2), activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv8)

    up9 = Conv2D(32, (2,2), activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv9)
    
    conv10 = Conv2D(3, (3, 3), activation = 'relu', padding = 'same')(conv9)
    return conv10

i = Input(shape=(256, 256,3))
o = InstantiateModel(i)
enhancer = Model(inputs=i, outputs=o)
enhancer.compile(optimizer="adam", loss='mean_squared_error')
enhancer.summary()

def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,256, 256,3)
        y_input = y[i].reshape(1,256, 256,3)
        yield (X_input,y_input)
enhancer.fit_generator(GenerateInputs(a,b),epochs=100,verbose=1,steps_per_epoch=20,shuffle=True)

enhancer.save_weights("C:/Users/gupta.LAPTOP-F7P9M08K/Desktop/modelx1.h5")
