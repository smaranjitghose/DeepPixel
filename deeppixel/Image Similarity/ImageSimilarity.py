# Step1: Import all the required librairies
from skimage.measure import compare_ssim
import argparse
import cv2
import imutils 

# Step2: Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first input image")
ap.add_argument("-s", "--second", required=True, help="second")
args = vars(ap.parse_args())

# Step3: Load the two input images and save them as image1 and image2
image1 = cv2.imread(args["first"])
image2 = cv2.imread(args["second"])

# Step4: Convert the images to grayscale as so to make the computation easy
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Step5: Compute the Structural Similarity Index (SSIM) between the two images
(score, diff) = compare_ssim(gray1, gray2, full=True)
diff = (diff * 255).astype("uint8")
print("the score is SSIM: {}".format(score))

if score == 1:
    print("The images are identical")
elif 1 > score > 0.95:
    print("The images are very similar")
else:
    print("The images are not the same. The difference between the 2 images is -  ", 1 - score)
