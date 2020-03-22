# Face detection
Face detection using Haar cascades is a machine learning based approach where a cascade function is trained with a set of input data. OpenCV already contains many pre-trained classifiers for face, eyes, smiles, etc. 

**AIM:** 

Take an image as input and detect the faces in it. A rectangle is drawn over the detected faces and they are saved as output

![face_detectino](https://user-images.githubusercontent.com/43414928/77245072-cf035e80-6c41-11ea-8ca5-fae9bef63099.png)


## Approach :
1. Import all the required libraries
2. Convert the input image into gray scale as haarcascade classifier accepts grayscale images
3. Import haarcascade classifier
4. Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). 
5. Once we get these locations, draw a rectangle or any other shape around the face
6. Save the output image


## Required libraries
1. OpenCV 
2. haarcascade Face classifier 
3. numpy 
4. matplotlib


## Reference
1. [Haar Cascade Object Detection Face & Eye OpenCV Python Tutorial ](https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/)
2. [Face Detection using Haar Cascades](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html)