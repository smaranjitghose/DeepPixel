# Detection of Faces in image

The aim is to create a script to detect faces in the input image as well as count the number of faces detected and saving output images with faces detected 

# Approach

OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. We will be using Face classifier for face detection . we need to load the required XML classifiers for detection.

**PROCESS :**
1) Import all the required libraries
2) Convert the input image into gray scale as haarcascade classifier accepts grayscale images
3) import haarcascade classifier
4) Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). 
5) Once we get these locations,  create a rectangle.
6) Save the output image

# Steps:

1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/face_rec/Face_detection


3) Install the requirements:
```
pip install -r requirements.txt 
```

4) Run the FaceDetection.py file with command:

```
python FaceDetection.py -i [IMAGE_PATH] -o [OUTPUT_FOLDER]

```
> for example
```
python FaceDetection.py -i Sample_images/run2.jpeg -o Sample_output/ 
```

This will save the image with faces marked in the output folder path provided.
