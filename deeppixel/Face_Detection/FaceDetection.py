 
from glob import glob 
import matplotlib.pyplot as plt
import cv2
import os
import argparse

#Function to convert RBG images to Gray
def convertToGray(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Function to convert gray images to RGB
def convertToRGB(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def faceDetection(img_path, output_folder, scaleFactor = 1.1):
	#OpenCV provides pre-trained models for detecting various objects like car , etc . here we are using haar cascade file for face detection only
	#Loading harcascade classifier for frontal face
	cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

	#Creating output folder if not present 
	if output_folder and not os.path.isdir(output_folder): 
	        os.mkdir(output_folder) 
	 	        
	image = cv2.imread(img_path)

	#convert the image to gray scale as opencv face detector expects gray images
	gray_image = convertToGray(image)
	   
	# Applying the haar classifier to detect faces
	faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)
	number_of_faces = len(faces_rect)
	
	for (x, y, w, h) in faces_rect:
	       cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 5)

	#convert to RGB
	img = convertToRGB(image)

	#Save the image to output folder
	if output_folder:
		input_filename, ext = os.path.splitext(os.path.basename(img_path))
		filename = os.path.join(output_folder, input_filename.format(ext))
		fig = plt.figure() 
		plt.imshow(img) 
		print("Faces found: ", number_of_faces)
		plt.axis("off")
		plt.savefig(filename,bbox_inches = 'tight', pad_inches = 0)
		plt.close(fig)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--img_path', type=str, help='Path to the input image')
	parser.add_argument('-o', '--output_folder', type=str, default=None, help='Path where the output image will be stored')
	args = parser.parse_args() 
	faceDetection(args.img_path, args.output_folder) 
	os._exit(0)


if __name__ == '__main__': 
    main()
