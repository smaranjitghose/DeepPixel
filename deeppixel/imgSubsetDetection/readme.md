# Image Subset Detection

The aim of this directory is to provide a methodology for performing the task of image subset detection. For example, given two images, we wish to find if one image is a subset of another image. For Example :
In the below, we have two images. Using the approach, we detected and highlighted the region where the cup (a subset) is present in the original image. 

 
 
### Approach

1.Load the input image and a image patch (the subset)
2.Convert the image patch (the subset) to grayscale.
3.To perform the task of match we use template matching function present in OpenCV function. 
4.Determin a threshold and then the template slides over the actual image and find the location where accuracy level matches.
5.Localize the location with higher matching probability
5.Higlight the area with highest matching probability by drawing a bounding box around it. 
 

### Reference 

1. [OpenCV documentation](https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html)


