# Motion Detection


## Aim

To detect motion in videos captured using webcam and enclose objects in motion in rectangles.


## Approach

1.Read back-to-back frames(images) from video.

2.Compute the absolute difference between the frames.

3.Convert color image to Gray scale and then to Gaussian blur so that change can be found easily.

4.If difference in pixel value is greater than 20, it is assigned white otherwise black.Dilate the thresholded image to fill in holes

5.Dilate the thresholded image to fill in holes then find contours on thresholded image.

6.If contour is too small ignore it else compute the bounding box for the contour, draw it on the frame.

## Output

## Reference

1.https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
 
