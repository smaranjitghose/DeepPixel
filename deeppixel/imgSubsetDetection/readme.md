# Image Subset Detection

The aim of this directory is to provide a methodology for performing the task of image subset detection. We can apporach this with two ways :  1. Single Instance Detection 2. Multiple Instance Detection

1. **Single Instance Detection** : In this task, we determine the position of the subset in the image for a single time (single instance). For example, in the below, we have two images. Using the approach, we detected and highlighted the region where the cup (a subset) is present in the original image. 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Original Image**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Image Subset**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Result Image**

&emsp;&emsp;&emsp;<img src="https://github.com/purva98/DeepPixel/blob/img_subset/deeppixel/imgSubsetDetection/images/input/cups.jpg" title="Original Image" width="250" height="350"/>&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/purva98/DeepPixel/blob/img_subset/deeppixel/imgSubsetDetection/images/input/subset_cup.jpg" alt="Your image title" width="120" height="120"/>&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/purva98/DeepPixel/blob/img_subset/deeppixel/imgSubsetDetection/images/output/cups_output.png" alt="Your image title" width="300" height="350"/> 




2. **Multiple Instance Detection** : In this task, we determine the position of the subset the image for more than one instances if they are present. The below examples will help us get better understanding. For example  :

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Original Image**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Image Subset**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Result Image**

&emsp;<img src="https://github.com/purva98/DeepPixel/blob/img_subset/deeppixel/imgSubsetDetection/images/input/cars.jpg" title="Original Image" width="300" height="350"/>&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/purva98/DeepPixel/blob/img_subset/deeppixel/imgSubsetDetection/images/input/subset_car.jpg" alt="Your image title" width="100" height="70"/>&emsp;&emsp;&emsp;&emsp;<img src="https://github.com/purva98/DeepPixel/blob/img_subset/deeppixel/imgSubsetDetection/images/output/car_output.png" alt="Your image title" width="290" height="350"/> 
 
 
### Approach

1.Load the input image and a image patch (the subset).


2.Convert the image patch (the subset) to grayscale.


3.To perform the task of match we use template matching function present in OpenCV function. 


4.After determining the threshold, the template slides over the actual image and find the location where accuracy level matches.


5.Localize the area with higher matching probability.


5.Higlight the area with highest matching probability by drawing a bounding box around it. 
 

### Reference 

1. [OpenCV documentation](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)

