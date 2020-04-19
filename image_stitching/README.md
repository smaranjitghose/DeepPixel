# Image Stitching

Image stitiching with OpenCV and Python


### Approach

1. Load the images. 

2. Stitch them together into a panorama using OpenCV’s built-in cv2.createStitcher / cv2.Stitcher_create functions.

3. After adding a border,create a gray version of stitched  image and threshold the gray image.

4. Apply contour extraction, compute the bounding box of the largest contour (i.e., the outline of the panorama itself) and draw the bounding box. 

5. Create two copies of the mask obtained.The size of one is progressively reduced until there are no more foreground pixels  left in the other, to get the smallest rectangular mask that can fit into the largest rectangular region of the panorama.

6. Given the minimum inner rectangle, find contours, compute the bounding box and extract the ROI from the stitched  image.

7. Write the output stitched image to the disk and display it. 

 
 
 ### Input 
 
 ![Input 1](https://github.com/gayathri-venu/DeepPixel/blob/master/deeppixel/image_stitching/input/IMG_1786-2.jpg)
 ![Input 2](https://github.com/gayathri-venu/DeepPixel/blob/master/deeppixel/image_stitching/input/IMG_1787-2.jpg)
 ![Input 3](https://github.com/gayathri-venu/DeepPixel/blob/master/deeppixel/image_stitching/input/IMG_1788-2.jpg)
  
 
 
 ### Output
 
 ![Output](https://github.com/gayathri-venu/DeepPixel/blob/master/deeppixel/image_stitching/output/output.png)
 
 
 
 ### References
 
 https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
 
 
    
