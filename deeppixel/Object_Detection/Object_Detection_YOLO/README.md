## Object Detection 

### Aim:
To detect objects in Image, video or through real-time webcam.

### Algorithm & Approach:
You Only Look Once or more popularly known as YOLO is one of the fastest real-time object detection algorithm (45 frames per seconds) as compared to R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN, etc.)

Link - https://pjreddie.com/darknet/yolo/

Here, we have used pre-trained model to detect objects in images, videos and real-time webcam.

### Steps:
1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/Object_Detection/Object_Detection_YOLO


3) Install the requirements:
```
pip install -r requirements.txt 
```
#### For Object Detection in Image

4) Run the ObjectDetectImage.py file with command:

```
python ObjectDetectImage.py -i [IMAGE_PATH] -o [OUTPUT_FOLDER]

```
> for example
```
python ObjectDetectImage.py -i asset/input/image/apple.jpeg -o asset/output/ 
```
This will save the image with objects marked and labeled in the output folder path provided.


#### For Object Detection using Webcam
*prerequisites:* Webcam

4) Run the ObjectDetectWebcam.py file with command:

```
python ObjectDetectWebcam.py 
```
> for example
```
python ObjectDetectWebcam.py 
```
This will open the webcam and label the detected objects

#### For Object Detection in Video

4) Run the ObjectDetectVideo.py file with command:

```
python ObjectDetectImage.py -v [VIDEO_PATH]

```
> for example
```
python ObjectDetectImage.py -v asset/input/video/car_on_road.mp4
```
This will open the video and label the detected objects


#### References 
1. [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
2. [Object Detection with OpenCV-Python using YOLOv3](https://medium.com/analytics-vidhya/object-detection-with-opencv-python-using-yolov3-481f02c6aa35)
3. [Object Detection using YoloV3 and OpenCV](https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420)


