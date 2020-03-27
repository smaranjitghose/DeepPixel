# Instance Segmentation
Instance segmentation is a subtype of image segmentation which identifies each instance of each object within the image at the pixel level. Instance segmentation, along with semantic segmentation, is one of two granularity levels of image segmentation.
Mask RCNN is the approach for instance segmentation. 
Instance Segmentaion have two process involved :
 *  Object Detection using Fast RCNN
 *  Semantic Segmentation
 
 In 2018 , FaceBookAI Research proposed a [research paper](https://arxiv.org/pdf/1703.06870.pdf) for the approach .
 
 The Segmented Image Examples are shown below:
 
 ### Input Image: 
 
 ![image](
https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Instance%20Segmentation/Input%20images/3651581213_f81963d1dd_z.jpg?raw=true
)

![iamge](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Instance%20Segmentation/Input%20images/4410436637_7b0ca36ee7_z.jpg?raw=true
)
 
 ### output image :
 
 ![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Instance%20Segmentation/Output%20Images/11.png?raw=true)

 ![image](https://github.com/Shweta0002/DeepPixel/blob/master/deeppixel/Instance%20Segmentation/Output%20Images/15.png?raw=true
)


## Approach
Overview on how to install
* Step 1: create a conda virtual environment with python 3.6
* Step 2: install the dependencies
* Step 3: Clone the Mask_RCNN repo
* Step 4: install pycocotools
* Step 5: download the pre-trained weights
* Step 6: Test it

Took reference from:
https://github.com/matterport/Mask_RCNN

watch a video on it https://www.youtube.com/watch?v=g7z4mkfRjI4

https://www.youtube.com/watch?v=5ZStcy7NWqs

## Requirements:
* numpy
* scipy
* cython
* h5py
* Pillow
* scikit-image, 
* tensorflow-gpu==1.5
* keras
