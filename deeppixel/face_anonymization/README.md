## Face Anonymization (Face Blurring)

Scripts to blur faces in images, videos and real time through webcam.

### Approach

1. Detection of faces
2. Extract face ROI
3. Blur/anonymize the face
4. Store blurred face in original image

### Steps:
1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/face_anonymization


#### For Face Anonymization in Image

3) Run the faceAnonymizationImage.py file with command:

```
python faceAnonymizationImage.py -i [IMAGE_PATH] -o [OUTPUT_FOLDER]

```
> for example
```
python faceAnonymizationImage.py -i asset/input/input.jpg -o asset/output/ 
```
This will save the image with pixalated faces in the output folder path provided.

##### Output Image
![Output Image](https://github.com/jhalak27/DeepPixel/blob/faceanonymization/deeppixel/face_anonymization/asset/output/input.png)


#### For Face Anonymization using Webcam
*prerequisites:* Webcam

3) Run the faceAnonymizationWebcam.py  file with command:

```
python faceAnonymizationWebcam.py 
```

This will open the webcam and pixalate the detected faces.

#### For Face Anonymization in Video

3) Run the faceAnonymizationVideo.py file with command:

```
python faceAnonymizationVideo.py -v [VIDEO_PATH]

```

This will open the video and pixalate the detected faces.

### References

[Blur and anonymize faces with OpenCV and Python](https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/)

