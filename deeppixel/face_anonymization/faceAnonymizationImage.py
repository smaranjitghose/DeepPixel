import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt

def loadCaffeModels():
	face_net = cv2.dnn.readNet('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
	return face_net


def loadImage(img_path):
	# image loading
	img = cv2.imread(img_path)
	return img


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


def faceBlurImage(img_path, output_folder, conf_threshold, blocks):
	image = loadImage(img_path)
	orig = image.copy()
	(h, w) = orig.shape[:2]
	net = loadCaffeModels()
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

	#convert to RGB
	img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
	
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
	parser.add_argument('-b', '--blocks', type=int, default=20, help='# of blocks for the pixelated blurring method')
	parser.add_argument('-c', '--confidence', type=float, default=0.7, help='minimum probability to filter weak detections')
	args = parser.parse_args() 
	faceBlurImage(args.img_path, args.output_folder, args.confidence, args.blocks) 
	os._exit(0)


if __name__ == '__main__': 
    main()