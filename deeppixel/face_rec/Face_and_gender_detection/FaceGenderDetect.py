# import necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import argparse


def genderDectection(img_path, output_folder):
		
	#Creating output folder if not present 
	if output_folder and not os.path.isdir(output_folder): 
	        os.mkdir(output_folder) 
	# read input image 	        
	img = cv2.imread(img_path)

	# apply face detection
	face, conf = cv.detect_face(img)

	padding = 5

	# loop through detected faces
	for f in face:

	    (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
	    (endX,endY) = min(img.shape[1]-1, f[2]+padding), min(img.shape[0]-1, f[3]+padding)
	    
	    # draw rectangle over face
	    cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)

	    face_crop = np.copy(img[startY:endY, startX:endX])

	    # apply gender detection
	    (label, confidence) = cv.detect_gender(face_crop)

	    idx = np.argmax(confidence)
	    label = label[idx]
	    label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
	    Y = startY - 5 if startY - 5 > 5 else startY + 5
	    cv2.putText(img, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
	                0.6, (0, 255, 255), 1)

	    
	# converting back to RBG    
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	#Save the image to output folder
	if output_folder:
		input_filename, ext = os.path.splitext(os.path.basename(img_path))
		filename = os.path.join(output_folder, input_filename.format(ext))
		fig = plt.figure() 
		plt.imshow(img) 
		plt.axis("off")
		plt.savefig(filename,bbox_inches = 'tight', pad_inches = 0)
		plt.close(fig)

	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--img_path', type=str, help='Path to the input image')
	parser.add_argument('-o', '--output_folder', type=str, default=None, help='Path where the output image will be stored')
	args = parser.parse_args() 
	genderDectection(args.img_path, args.output_folder) 
	os._exit(0)


if __name__ == '__main__': 
    main()