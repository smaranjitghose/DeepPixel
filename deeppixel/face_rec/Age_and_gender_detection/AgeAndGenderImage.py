from glob import glob 
import cv2
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import os

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    face_net = cv2.dnn.readNet('opencv_face_detector.pbtxt', 'opencv_face_detector_uint8.pb')
    age_net = cv2.dnn.readNet('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNet('deploy_gender.prototxt', 'gender_net.caffemodel')
    return face_net, age_net, gender_net

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def loadImage(img_path):
	# image loading
	img = cv2.imread(img_path)
	return img


def imageAgeAndGenderDetection(img_path, output_folder): 

	face_net, age_net, gender_net = load_caffe_models()

	image = loadImage(img_path)

	padding=20
	resultImg,faceBoxes = highlightFace(face_net,image)

	if not faceBoxes:
	    print("No face detected")

	else:
	    print("Found {} faces".format(str(len(faceBoxes))))


	for faceBox in faceBoxes:
	    face=image[max(0,faceBox[1]-padding):
	               min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
	               :min(faceBox[2]+padding, image.shape[1]-1)]

	    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

	    #Predict Gender
	    gender_net.setInput(blob)
	    gender_preds=gender_net.forward()
	    gender=gender_list[gender_preds[0].argmax()]
	    print("Gender : " + gender)

	    #Predict Age
	    age_net.setInput(blob)
	    age_preds = age_net.forward()
	    age = age_list[age_preds[0].argmax()]
	    print("Age Range: " + age)
	    overlay_text = "%s %s" % (gender, age)
	    cv2.putText(resultImg, overlay_text, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2, cv2.LINE_AA)

	
	#convert to RGB
	img = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
	
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
	imageAgeAndGenderDetection(args.img_path, args.output_folder) 
	os._exit(0)


if __name__ == '__main__': 
    main()