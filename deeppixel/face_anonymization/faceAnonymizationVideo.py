import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

def loadCaffeModels():
	face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
	return face_net


def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)

    # return the pixelated blurred image
    return image


def faceBlurImage(video_path, conf_threshold, blocks):
	cap = cv2.VideoCapture(video_path)
	
	net = loadCaffeModels()

	while cv2.waitKey(1)<0:
		hasFrame,frame=cap.read()

		if not hasFrame:
			break
		
		orig = frame.copy()
		(h, w) = orig.shape[:2]
		# construct a blob from the image
		blob = cv2.dnn.blobFromImage(orig, 1.0, (300, 300), (104.0, 177.0, 123.0))

		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > conf_threshold:
				startX=int(detections[0,0,i,3]*w)
				startY=int(detections[0,0,i,4]*h)
				endX=int(detections[0,0,i,5]*w)
				endY=int(detections[0,0,i,6]*h)
				face = orig[startY:endY, startX:endX]
				face = anonymize_face_pixelate(face, blocks=blocks)

				# store the blurred face in the output image
				orig[startY:endY, startX:endX] = face

		cv2.imshow("Video", orig)
		# Quits when 'Q' or 'q' or 'esc' is pressed 
		ch = cv2.waitKey(1)
		if ch == 27 or ch == ord('q') or ch == ord('Q'):
		    break

	cap.release()
	cv2.destroyAllWindows()



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--video_path', type=str, help='Path to the input video')
	parser.add_argument('-b', '--blocks', type=int, default=20, help='# of blocks for the pixelated blurring method')
	parser.add_argument('-c', '--confidence', type=float, default=0.7, help='minimum probability to filter weak detections')
	args = parser.parse_args() 
	faceBlurImage(args.video_path, args.confidence, args.blocks) 
	os._exit(0)


if __name__ == '__main__': 
    main()