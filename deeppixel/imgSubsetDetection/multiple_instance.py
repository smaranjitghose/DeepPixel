# Importing necessary libraries
import cv2
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-original", "--original", required=True, help="first input image")
ap.add_argument("-subset", "--subset", required=True, help="second input image")
args = vars(ap.parse_args())

#Load the two input images and save them as larger image and smaller images
large_image = cv2.imread(args["original"])
large_image_org = cv2.imread(args["original"])
img_gray = cv2.imread(args["original"],0)
small_image = cv2.imread(args["subset"])
small_image = cv2.imread(args["subset"])
subset_gray = cv2.imread(args["subset"],0)
width, height = subset_gray.shape[::-1]


# Use OpenCv matching Template 
result_image = cv2.matchTemplate(img_gray,subset_gray,cv2.TM_CCOEFF_NORMED)
thresh = 0.8
location = np.where(result_image >= thresh)
for i in zip(*location[::-1]):
    cv2.rectangle(large_image,i,(i[0] + width, i[1] + height), (0,0,255), 2)

#Displaying the result
cv2.imshow("Original image",large_image_org)
cv2.imshow("With subset detected", large_image)
cv2.waitKey(0)