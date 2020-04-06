from glob import glob 
import cv2
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import os


def loadYolo():
	net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	# colors = np.random.uniform(0, 255, size=(len(classes), 3))
	colors = (0, 255, 255)
	return net, classes, colors, output_layers

def loadImage(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels

def detectObjects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def getBoxDimensions(outputs, height, width):
	class_ids=[]
	confidences=[]
	boxes=[]
	for out in outputs:
	    for detection in out:
	        scores = detection[5:]
	        class_id = np.argmax(scores)
	        confidence = scores[class_id]
	        if confidence > 0.5:
	            #onject detected
	            center_x= int(detection[0]*width)
	            center_y= int(detection[1]*height)
	            w = int(detection[2]*width)
	            h = int(detection[3]*height)
	        
	            
	            #rectangle co-ordinaters
	            x=int(center_x - w/2)
	            y=int(center_y - h/2)
	            
	            boxes.append([x,y,w,h]) #put all rectangle areas
	            confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
	            class_ids.append(class_id) #name of the object tha was detected

	return boxes, confidences, class_ids
			
def drawLabels(boxes, confidences, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
	    if i in indexes:
	        x, y, w, h = boxes[i]
	        label = str(classes[class_ids[i]])
	        # color = colors[i]
	        cv2.rectangle(img, (x,y), (x+w, y+h), colors, 2)
	        cv2.putText(img, label, (x, y - 5), font, 1.5, colors, 1)
	return img

def imageObjectDetection(img_path, output_folder): 

	#Creating output folder if not present 
	if output_folder and not os.path.isdir(output_folder): 
	        os.mkdir(output_folder) 


	model, classes, colors, output_layers = loadYolo()
	image, height, width, channels = loadImage(img_path)
	blob, outputs = detectObjects(image, model, output_layers)
	boxes, confs, class_ids = getBoxDimensions(outputs, height, width)
	img = drawLabels(boxes, confs, colors, class_ids, classes, image)
	
	#convert to RGB
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
	imageObjectDetection(args.img_path, args.output_folder) 
	os._exit(0)


if __name__ == '__main__': 
    main()