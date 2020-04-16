# Step1 : Import required libraries
from PIL import Image
import pytesseract
import numpy as np
import argparse
import cv2, os

# Step 2: Parse the argument
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required = True)
parser.add_argument("-p", "--preprocess", type = str, default = "thresh")
args = vars(parser.parse_args())

# Step 3: Load the  image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Step 4: Check preprocess to apply thresholding on the image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)
 
# Step 5: Write the grayscale image to disk as a temporary file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# Step 6: Load the image as a PIL/Pillow image and apply OCR
text = pytesseract.image_to_string(Image.open(filename))

# Step 7: Display the text
print("the text is \n", text)
 
# Step 8: Show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()