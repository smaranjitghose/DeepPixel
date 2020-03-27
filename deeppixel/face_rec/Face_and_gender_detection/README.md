## Detection of Faces in image

A script to detect faces and gender in the input image given and saves labelled output images withassociated probabilities in given folder 

### Approach

Uses cvlib which is open source Computer Vision library for Python. Underneath cvlib is using an AlexNet-like model trained on Adience dataset by Gil Levi and Tal Hassner for their CVPR 2015 paper.

### Steps

1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/face_rec/Face_and_gender_detection


3) Install the requirements:
```
pip install -r requirements.txt 
```

4) Run the faceGenderDetect.py file with command:

```
python faceGenderDetect.py -i [IMAGE_PATH] -o [OUTPUT_FOLDER]

```
> for example
```
python faceGenderDetect.py -i asset/input/input.jpg -o asset/output/ 
```

This will save the image with objects labelled in the output folder path provided.

### References

[cvlib](https://www.cvlib.net/)
