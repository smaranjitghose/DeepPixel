# Semantic Segmentation


Semantic segmentation algorithm attempts to:

1. Partition the image into meaningful parts
2. While at the same time, associate every pixel in an input image with a class label (i.e., person, road, car, bus, etc.)


### Aim

To implement Semantic segmentation on images and videos using OpenCV and deep learning.


### Approach


#### Semantic segmentation on images

1. Load the serialized model from disk.The semantic segmentation architecture we’re using here is ENet and it is trained on The Cityscapes Dataset, a semantic, instance-wise, dense pixel annotation of 20-30 classes

2. Construct a blob so that the input images have a resolution of 1024 x 512.

3. Set the blob  as input to the network and perform a forward pass through the neural network.

4. Extract volume dimension information from the output of the neural network.

5. Generate classMap  by finding the class label index with the largest probability for each and every pixel of the output  image array

6. Compute color mask  from the colors associated with each class label index in the classMap

7. Resize the mask to match the input image dimensions.

8. And finally, overlay the mask on the frame transparently to visualise the segmentation.


#### Semantic segmentation on videos

1. Load the serialized model from disk.

2. Open a video stream pointer to input video file on and initialize our video writer object.

3. Loop over the frames of the video and perform the same steps we performed for images.

4. Write the output frame to disk. 


### Output

![Output1](https://github.com/gayathri-venu/Semantic_segmentation/blob/master/output/output1.png)

![Output2](https://github.com/gayathri-venu/Semantic_segmentation/blob/master/output/output2.png)


### Reference

https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/
