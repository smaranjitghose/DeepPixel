## Detection of Faces in image

A script to detect objects in the input image given and saves marked and labelled output images in given folder 

### Approach

Uses cvlib which is open source Computer Vision library for Python. Underneath it uses YOLOv3 model trained on COCO dataset capable of detecting [80 common objects](https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt) in context.

### Steps

1) Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```

2) Navigate to the directory deeppixel/Object_Detection


3) Install the requirements:
```
pip install -r requirements.txt 
```

4) Run the ObjectDetection.py file with command:

```
python ObjectDetection.py -i [IMAGE_PATH] -o [OUTPUT_FOLDER]

```
> for example
```
python ObjectDetection.py -i asset/input/image.jpg -o asset/output/ 
```

This will save the image with objects marked and labelled in the output folder path provided.

### References

[cvlib](https://www.cvlib.net/)
