from deeppixel.edge_detection.canny_edge import image_edge_detect,visualize
from deeppixel.edge_detection.hed import image_to_HED
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", type=str, required=True,
	help="path to the image file/folder")
ap.add_argument("-k", "--kernel-size", type=int, required=False,
	help="size of the kernel")
ap.add_argument("-w", "--weak-pixel", type=int, required=False,
	help="value of the weaker pixel")
ap.add_argument("-s", "--strong-pixel", type=int, required=False,
	help="value of the stronger pixel")
ap.add_argument("-lt", "--low-threshold", type=float, required=False,
	help="value of low-threshold for the image")
ap.add_argument("-ht", "--high-threshold", type=float, required=False,
	help="value of higher-threshold for the image")
args = vars(ap.parse_args())

img=image_edge_detect(args["path"])
img2=image_to_HED(args["path"])

visualize(img)
visualize(img2)
plt.show()