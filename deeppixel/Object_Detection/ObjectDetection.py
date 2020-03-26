import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os
import argparse


def objectDetection(img_path, output_folder):
	#Creating output folder if not present 
	if output_folder and not os.path.isdir(output_folder): 
	        os.mkdir(output_folder) 
	 	        
	im = plt.imread(img_path)
	
	bbox, label, conf = cv.detect_common_objects(im)
	output_image = draw_bbox(im, bbox, label, conf)
	

	#Save the image to output folder
	if output_folder:
		input_filename, ext = os.path.splitext(os.path.basename(img_path))
		filename = os.path.join(output_folder, input_filename.format(ext))
		fig = plt.figure() 
		plt.imshow(output_image) 
		plt.show()
		plt.axis("off")
		plt.savefig(filename,bbox_inches = 'tight', pad_inches = 0)
		plt.close(fig)



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--img_path', type=str, help='Path to the input image')
	parser.add_argument('-o', '--output_folder', type=str, default=None, help='Path where the output image will be stored')
	args = parser.parse_args() 
	objectDetection(args.img_path, args.output_folder) 
	os._exit(0)


if __name__ == '__main__': 
    main()