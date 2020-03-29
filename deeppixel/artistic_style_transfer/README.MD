

# Style Transfer with Deep Neural Networks

Neural style transfer is a technique which is used to blend together two images - a content image and a style reference image (such as an artwork by a famous painter) seamlessly into one

**AIM**
The aim of this task is to transfer the style of the reference image to the content image 

![styleTransfer](https://user-images.githubusercontent.com/43414928/77195656-00780f00-6b08-11ea-87a3-434759253954.jpeg)

### Approach
1. We take a input image (ie a content image ) and style images and resize them to equal shapes.
2. We load VGG19, pre-trained Convolutional Neural Network and create the model. 
3. Next we load in the content and style image and pre process them
4. Then we set up the loss function 
   * content loss (distance between the input and output images - we strive to preserve the content)
   * style loss (distance between the style and output images - we strive to apply a new style)
   * total variation loss (some contstant * style loss + some contstant * content loss)
5. Finally, we optimize for loss function


### How to run the code

1. If you would like to test how what the output image looks like, run the following code in the terminal:
python style_transfer_using_Pytorch.py

2. If you would like to run the code on content and style images chosen by you, run the following code in the terminal:
python style_transfer_using_Pytorch.py --path_to_content_image 'absolute path of content image' --path_to_style_image 'absolute path of style image' --path_to_save_output_file 'absolute path of location where you would like to save the image'



### Input 
![Screenshot (2)](https://user-images.githubusercontent.com/43414928/77197522-57cbae80-6b0b-11ea-951e-eb8f5f1811e3.png)


### Output
![Screenshot (3)](https://user-images.githubusercontent.com/43414928/77197527-58644500-6b0b-11ea-962b-4ff9c5ad3d91.png)

### Required libraries
1. pytorch
2. matplotlib
3. numpy


### Reference:

1. A Neural Algorithm of Artistic Style Paper [here](https://arxiv.org/pdf/1508.06576.pdf)
2. Official Pytorch Neural transfer tutorial documentation [here](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html?highlight=vgg)
 
