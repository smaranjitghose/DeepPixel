import argparse
import brisque as b

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path of the Image that'll be used")
ap.add_argument("-b", "--blur", required=False, help="Type True to get Blur scores else False")
args = vars(ap.parse_args())

img, gray = b.imgread(args["path"])
b.imgscore(img,args["blur"])
b.imgshow(img)