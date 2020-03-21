# Detection of Faces in image

A script to detect faces in the input image as well as count the number of faces detected and saving output images with faces detected 

# Approach

OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. We will be using Face classifier for face detection. We need to load the required XML classifiers for detection.

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
