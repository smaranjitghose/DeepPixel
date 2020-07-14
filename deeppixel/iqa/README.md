# This module(brisque) deals with:
- Assessing the Image Quality Scores 
- Blur_Detection in the images that we upload/refer a path to.

**To import this module simple type:**
from deeppixel import iqa.brisque as b

**To Read an Image from local computer:**
img_color,img_gray = b.imgread( #path to the image file)
*Note that It returns 2 images, one is color(RGB) and the other one is grayscale.*

**To Read an Image from URL:**
img_color,img_gray = b.imgurl( #URL to the image file)
*Note that It returns 2 images, one is color(RGB) and the other one is grayscale.*

**To Preview the Image:**
b.imgshow( #image returned by b.imgread/b.imgurl,  #title)
*Note that Title can be anything of your choice

**To Get the Image Assessment and Blur Scores**
b.imgscore( #image returned by b.imgread, #True or False)
*Note For True,we will get the Blur Detection scores as well. But for False, we will get back only the Image assessment scores*

## USING ARGPARSE:
There is another file created called user_argparse.py
This file mainly accesses the command line of your system to get the path the Image as input

```terminal
cd to the iqa folder(!cd iqa)
```
```terminal
Python user_argparse.py --path "<path to your Image file>" --blur True/False
```
- True/False is optional, but its an value just to check if the user wants the Blur scores or not.
- The Results are printed
- Picture is Displayed.

## TEST RESULTS FROM TEST.PY

**Note:Lower the Assessed Image Scores, Better is the Picture
         Lower the Blur Detection score, Blurrier is the Picture**

1.HD.jpg
```terminal
   -Blur Detection Results (Scores) :  Not Blurry (34.8772)
   -Assessed Image has a score of :  11.7968082850617
```

2.Noise.png
```terminal
   -Blur Detection Results (Scores) :  Not Blurry (63.1739)
   -Assessed Image has a score of :  155.845
```

3.Blur.png
```terminal
   -Blur Detection Results (Scores) :  Blurry (-8.9895)
   -Assessed Image has a score of :  81.75123706366387
```
 

## BIBLIOGRAPHY/REFERENCES(random order):
1.https://www.learnopencv.com/image-quality-assessment-brisque/
2.https://in.mathworks.com/help/images/ref/brisque.html
3.http://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf
4.http://live.ece.utexas.edu/publications/2011/am_asilomar_2011.pdf





 