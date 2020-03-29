# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", required=True,
	help="path to output video file")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
	help="desired width (in pixels) of input image")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")

# if a colors file was supplied, load it from disk
if args["colors"]:
	COLORS = open(args["colors"]).read().strip().split("\n")
	COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
	COLORS = np.array(COLORS, dtype="uint8")

# otherwise, we need to randomly generate RGB colors for each class
# label
else:
	# initialize a list of colors to represent each class label in
	# the mask (starting with 'black' for the background/unlabeled
	# regions)
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
		dtype="uint8")
	COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

# load our serialized model from disk
net = cv2.dnn.readNet(args["model"])

# initialize the video stream and pointer to output video file
vs = cv2.VideoCapture(args["video"])
writer = None


# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# construct a blob from the frame and perform a forward pass
	# using the segmentation model
	frame = imutils.resize(frame, width=args["width"])
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (1024, 512), 0,
		swapRB=True, crop=False)
	net.setInput(blob)
	output = net.forward()
	

	# infer the total number of classes along with the spatial
	# dimensions of the mask image via the shape of the output array
	(numClasses, height, width) = output.shape[1:4]

	# our output class ID map will be num_classes x height x width in
	# size, so we take the argmax to find the class label with the
	# largest probability for each and every (x, y)-coordinate in the
	# image
	classMap = np.argmax(output[0], axis=0)

	# given the class ID map, we can map each of the class IDs to its
	# corresponding color
	mask = COLORS[classMap]

	# resize the mask such that its dimensions match the original size
	# of the input frame
	mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
		interpolation=cv2.INTER_NEAREST)

	# perform a weighted combination of the input frame with the mask
	# to form an output visualization
	output = ((0.3 * frame) + (0.7 * mask)).astype("uint8")

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(output.shape[1], output.shape[0]), True)



	# write the output frame to disk
	writer.write(output)
        
	#display the output frame
	cv2.imshow("Frame", output)
	key = cv2.waitKey(1) & 0xFF
 
	# if the esc key was pressed, break from the loop
	if key == 27:
	       break

# release the file pointers
writer.release()
vs.release()

# USAGE
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/massachusetts.mp4 --output output/massachusetts_output.avi
# python segment_video.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --video videos/toronto.mp4 --output output/toronto_output.avi
