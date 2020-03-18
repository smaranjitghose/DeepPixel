from glob import glob 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

def colorize(input_path, input_format, output_folder):
    print("----->Defining the model paths")
    #For this task we will use a trained model(which is available publically)
    v2_prototxt = "model/colorization_deploy_v2.prototxt"
    caffemodel = "model/colorization_release_v2.caffemodel"
    points = "model/pts_in_hull.npy"

    InputPath= input_path
    InputFormat= '*.'+input_format
    OutputPath = output_folder  
   
    # We load the serialized black and white colorizer 
    net = cv2.dnn.readNetFromCaffe(v2_prototxt,caffemodel)
    points = np.load(points)
    # Now we add the cluster centres to the model as 1*1 convolutions
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    images = sorted(glob(os.path.join(InputPath,InputFormat)))  
    if OutputPath and not os.path.isdir(OutputPath): 
        os.mkdir(OutputPath) 
    o_formats = InputFormat.split(',')
    
    for i, img_path in enumerate(images): 
            print("----->Processing image {} / {}".format(i+1,len(images)))    
            
            # We load the input image and convert it to gray scale just to clean them.
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Next our image needs to be converted from rgb to Lab
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # Before converting to Lab, we scale in order to extract the 'L' channel and centre it in future 
            scaled = image.astype("float32") / 255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
            resized_image = cv2.resize(lab, (224, 224))
            L_channel = cv2.split(resized_image)[0]
            L_channel -= 50            
            net.setInput(cv2.dnn.blobFromImage(L_channel))
            a_and_b = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(a_and_b, (image.shape[1], image.shape[0]))
            L = cv2.split(lab)[0]
            # After getting the predicted values of 'a' and 'b' we combine the L+a+b
            LAB_colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            # For better visualisation we covert our Lab image to rgb image
            lab_to_rgb_colorized = cv2.cvtColor(LAB_colorized, cv2.COLOR_LAB2RGB)
            clipped_colorized = np.clip(lab_to_rgb_colorized, 0, 1)
            colorized = (255 * clipped_colorized).astype("uint8")
          
            if output_folder:
                input_filename, ext = os.path.splitext(os.path.basename(img_path))
                filename = os.path.join(OutputPath, input_filename.format(ext))
                fig = plt.figure() 
                ii = plt.imshow(colorized, interpolation='nearest') 
                plt.axis("off")
                plt.savefig(filename,bbox_inches = 'tight', pad_inches = 0)
                plt.close(fig)
    print("----->Saved all the images") 
   
                
def main():
    print("----->Importing the required libraries")
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path where the input images are stored')
    parser.add_argument('-f', '--input_format', type=str, default="jpg",help='The input format: jpg or png')
    parser.add_argument('-o', '--output_folder', type=str, default=None,help='Path where the output images will be stored')
    args = parser.parse_args() 
    colorize(args.input_path,args.input_format, args.output_folder) 
    os._exit(0)

if __name__ == '__main__': 
    main()

