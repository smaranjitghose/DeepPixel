from glob import glob 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import cv2
import os
import argparse


def bright(input_image):
            threshold=26  
            #converting image to gray scale using openCV 
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            
            # Calculate grayscale histogram and find the cummulative distribution 
            histogram = cv2.calcHist([gray_image],[0],None,[256],[0,256])
            cummulative = []
            cummulative.append(float(histogram[0]))
            for i in range(1,len(histogram)):
                cummulative.append(cummulative[i-1] + float(histogram[i]))

            # We find the place in the histogram hwere the color frequency is less than the decided threshold 
            max_range = cummulative[-1]
            threshold *= (max_range/100.0)
            threshold /= 2.0

            # We cut the left side of the histogram where frequency is less than this threshold
            minimum_gray = 0
            while cummulative[minimum_gray] < threshold:
                    minimum_gray += 1

            # We cut the right side of the histogram where the frequency is greater than the threshold
            maximum_gray = len(histogram) -1
            while cummulative[maximum_gray] >= (max_range - threshold):
                    maximum_gray -= 1

            # Alpha is brightness, and Beta is the contrast 
            # Now we calculate alpha and beta values, so that they are in the range [0...255]
            alpha = 255 / (maximum_gray - minimum_gray)
            beta = -minimum_gray * alpha
    
            # Now with the adaptively generated Alph and Beta values we calculate the converted scale image
            converted_scale_img = input_image * alpha + beta
            converted_scale_img[converted_scale_img < 0] = 0
            converted_scale_img[ converted_scale_img > 255] = 255
            adaptive_image= converted_scale_img.astype(np.uint8) 
            
            return(adaptive_image)
            

    
def adaptive_brightness_contrast(input_path,output_path):
 
    data_dir_list = os.listdir(input_path)
    if output_path and not os.path.isdir(output_path): 
        os.mkdir(ouput_path) 
    
    for img_path in data_dir_list: 
            image_path=(input_path+img_path)
            print(image_path)
            input_image = Image.open(image_path)
            input_image = asarray(input_image)         
            adaptive_image= bright(input_image)
            if output_path:
                fig = plt.figure() 
                ii = plt.imshow(adaptive_image, interpolation='nearest') 
                plt.axis("off")
                plt.savefig(output_path+img_path,bbox_inches = 'tight', pad_inches = 0)
                plt.close(fig)
                
  
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_path',type=str, help='Path where the input images are stored')
    parser.add_argument('-o', '--output_path', type=str, default=None,help='Path where the output images will be stored')
    args = parser.parse_args() 
    adaptive_brightness_contrast(args.input_path, args.output_path) 
    os._exit(0)

if __name__ == '__main__': 
    main()

