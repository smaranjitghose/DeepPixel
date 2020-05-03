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
small_image = cv2.imread(args["subset"])

# Use OpenCv matching Template 
# We simply slide the smaller image over the larger image, and perform template matching
matched_area = cv2.matchTemplate(np.array(small_image), np.array(large_image), cv2.TM_SQDIFF_NORMED)

# Now we use minMaxLoc of OpenCV to extract the minimum squared diffence
minimum_1,_,minimum_f,_ = cv2.minMaxLoc(matched_area)
# Now we store the coordinates of our best template matched result
best_x,best_y= minimum_f

# No. of rows and columns
R,C = np.array(small_image).shape[:2]
# Drawing the rectangle on the larger image and displaying the results
result_image = cv2.rectangle(np.array(large_image), (best_x,best_y),(best_x+C,best_y+R),(0,0,255),2)

#Displaying the result
cv2.imshow("Original image",large_image)
cv2.imshow("With subset detected", result_image)
cv2.waitKey(0)