from deeppixel.iqa import brisque as b
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path of the Image that'll be used")
ap.add_argument("-b", "--blur", required=False, help="Type True to get Blur scores else False")
args = vars(ap.parse_args())

img, gray = b.imgread(args["path"])

iqa, val = b.imgscore(img, args["blur"])
if args["blur"] == "True":
    print("Blur Detection Results : ",val)
print("Image Quality Score : ",iqa)






