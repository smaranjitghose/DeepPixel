# WebCam Motion Detector

### Aim
  
To create a Webcame Motion Detector that allows you to detect motion and also store the time interval of the motion.

### Approach

Videos can be treated as stack of pictures called frames. 

We compare different frames(pictures) to the first frame which should be static(No movements initially).

We compare two images by comparing the intensity value of each pixels.

After running the code there 4 new window will appear on screen.

1. Gray Frame : In Gray frame the image is a bit blur and in grayscale we did so because, In gray pictures there is only one intensity value whereas in RGB(Red, Green and Blue) image there are three intensity values.
So it would be easy to calculate the intensity difference in grayscale.

2. Difference Frame : Difference frame shows the difference of intensities of first frame to the current frame.

3. Threshold Frame : If the intensity difference for a particular pixel is more than 30 then that pixel will be white and if the difference is less than 30 that pixel will be black

4. Color Frame : In this frame you can see the color images in color frame along with green contour around the moving objects

The Time_of_movements file will be stored in the folder where your code file is stored. This file will be in csv extension. In this file the start time of motion and the end time of motion will be recorded.

### Requirements

1.Python3

2.OpenCV

3.Pandas

### References

https://www.geeksforgeeks.org/webcam-motion-detector-python/
