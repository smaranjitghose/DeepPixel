## Real-Time Face Detection

A script to detect faces in video via a webcam in real time.

# Approach

OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. We will be using Face classifier for face detection. 

# Pre-requisites

A working webcam

# Steps

1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/face_rec/Face_detection_real_time


3) Install the requirements:
```
pip install -r requirements.txt 
```

4) Run the FaceDetectRealTime.py file with command:

```
python FaceDetectionRealTime.py 

```

This start your webcam and detect faces will have boundary boxes.
You can exit by pressing 'Esc' or 'Q' or 'q'. 
