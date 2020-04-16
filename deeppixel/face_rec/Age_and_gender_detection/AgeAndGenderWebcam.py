from glob import glob 
import cv2
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import os
from imutils.video import WebcamVideoStream

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


def webcamAgeAndGenderDetection(): 

	face_net, age_net, gender_net = load_caffe_models()
	video_capture = WebcamVideoStream(src=0).start()
	# image = loadImage(img_path)
	while True:
		padding=20
		frame = video_capture.read()
		resultImg,faceBoxes = highlightFace(face_net,frame)


		for faceBox in faceBoxes:
		    face=frame[max(0,faceBox[1]-padding):
		               min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
		               :min(faceBox[2]+padding, frame.shape[1]-1)]

		    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

		    #Predict Gender
		    gender_net.setInput(blob)
		    gender_preds=gender_net.forward()
		    gender=gender_list[gender_preds[0].argmax()]

		    #Predict Age
		    age_net.setInput(blob)
		    age_preds = age_net.forward()
		    age = age_list[age_preds[0].argmax()]
		    overlay_text = "%s %s" % (gender, age)
		    cv2.putText(resultImg, overlay_text, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2, cv2.LINE_AA)


		cv2.imshow("Video", resultImg)
		# Quits when 'Q' or 'q' or 'esc' is pressed 
		ch = cv2.waitKey(1)
		if ch == 27 or ch == ord('q') or ch == ord('Q'):
		    break

	video_capture.stop()
	cv2.destroyAllWindows() 

def main(): 
	webcamAgeAndGenderDetection() 
	os._exit(0)


if __name__ == '__main__': 
    main()