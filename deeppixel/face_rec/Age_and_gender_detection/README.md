## Age and Gender Detection of Faces

To detect age and gender of faces in the input image, video or real-time through webcam.

### Approach

 We have used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100).

Link: [Gil Levi and Tal Hassner.Age and Gender Classification using Convolutional Neural Networks](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf)

We have done as follow:
1. Detection of faces
2. Classification into Male/Female
3. Classification into one of the age group

### Steps:
1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/face_rec/Age_and_gender_detection


#### For Age And Gender Detection in Image

3) Run the AgeAndGenderImage.py file with command:

```
python AgeAndGenderImage.py -i [IMAGE_PATH] -o [OUTPUT_FOLDER]

```
> for example
```
python AgeAndGenderImage.py -i asset/input/woman.jpeg -o asset/output/ 
```
This will save the image with faces marked and labeled with age and gender in the output folder path provided.

##### Output Image
![Output Image](https://github.com/jhalak27/DeepPixel/blob/AgeGender/deeppixel/face_rec/Age_and_gender_detection/asset/output/woman.png)


#### For Age And Gender Detection using Webcam
*prerequisites:* Webcam

3) Run the AgeAndGenderWebcam.py  file with command:

```
python AgeAndGenderWebcam.py 
```

This will open the webcam and label the detected faces with gender and age

#### For Age And Gender Detection in Video

3) Run the AgeAndGenderVideo.py file with command:

```
python AgeAndGenderVideo.py -v [VIDEO_PATH]

```

This will open the video and label the detected faces with gender and age 

### References

[Tal Hassner](https://talhassner.github.io/home/publication/2015_CVPR)

[Gender and Age Detection with OpenCV](https://data-flair.training/blogs/python-project-gender-age-detection/)
