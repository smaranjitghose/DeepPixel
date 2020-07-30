# This module deals with Detecting Edges via :
- Canny Edge Detection(5 step) Algorithm
- Holistically-Nested edge Detection

**To import this module type**
from deeppixel import edge_detection.(canny_edge/hed)

## To Implement Canny Edge Detection
**For single image:**

from canny_edge import image_edge_detect

results=image_edge_detect(<path to the image>)

**To get the results folder wise:**

 from canny_edge import folder_edge_detect

 results=folder_edge_detect(<path to the image>)

*This returns a list of Images*

**To visualize the images:**

from canny_edge import visualize

visualize(image,format_image)

*Here format_image pertains to the colour of the image*

## To Implement Holistically Nested Edge Detection

**For single image:**

from hed import image_to_HED

results=image_to_HED(<path to the image>)
 
**To get the results folder wise:**

from hed import folder_to_HED 
results=folder_to_HED(<path to the image>)

*This returns a list of Images*

### BIBLIOGRAPHY/REFERENCES :

- https://en.wikipedia.org/wiki/Canny_edge_detector

- https://medium.com/@ssatyajitmaitra/what-canny-edge-detection-algorithm-is-all-about-103d94553d21

- http://fourier.eng.hmc.edu/e161/lectures/canny/node1.html

- http://justin-liang.com/tutorials/canny/

- https://arxiv.org/abs/1504.06375

-  https://www.pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/
