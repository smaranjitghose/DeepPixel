import matplotlib.pyplot as plt
import cv2
import glob
import argparse
import numpy as np
import model1
from model1 import *

ap = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

i = Input(shape=(256, 256,3))
o = InstantiateModel(i)
enhancer = Model(inputs=i, outputs=o)
enhancer.compile(optimizer="adam", loss='mean_squared_error')
enhancer.summary()

enhancer.load_weights("C:/Users/gupta.LAPTOP-F7P9M08K/Desktop/modelx1.h5")

ImagePath=cv2.imread(args['image'])
ImagePath=cv2.cvtColor(ImagePath, cv2.COLOR_BGR2RGB)
I=cv2.resize(ImagePath, ((256, 256)))
I = I.reshape(1, 256,256,3)
Prediction = enhancer.predict(I)
I=I.reshape(256, 256, 3)
I[:,:,:] = Prediction[:,:,:]
cv2.imwrite('output_images/predicted_1.png', I)

