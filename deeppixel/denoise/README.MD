# Denoising Images
Images often get corrupted because of various factors that can occur during acquisition and transmission processes. Image de-noising is the process of removing noise from an image, while at the same time preserving details and structures. 
**Aim**
The aim of this task is to generate a script which produces denoised images when a noisy image is provided as input.
![denoise](https://user-images.githubusercontent.com/43414928/77244677-b729db80-6c3d-11ea-825d-cc1285ba1497.png)


 
  ## Approach:
  The approach used is :
  1. An image with Gaussian noise is taken as input
  2. From the noisy image low frequency and high-frequency components are extracted using filtering techniques.
  3. Then the high and low-frequency components are combined in order to generate the denoised image.
  

 ## Required libraries
 1. skimage 
 2. matplotlib 
 3. numpy 
 4. cv2 

  ## Colab Link
  https://colab.research.google.com/drive/1I8SDSPGacrpol0k4-_6Wrk-6rcpem7ww
  
  ## References:
  1. [Self-Learning Based Image Decomposition With Applications to Single Image Denoising](https://www.researchgate.net/publication/264573776_Self-Learning_Based_Image_Decomposition_With_Applications_to_Single_Image_Denoising)
  2. [Single Image Super-Resolution Using Deep Learning](https://in.mathworks.com/help/images/single-image-super-resolution-using-deep-learning.html)
  3. [Image filters in python](https://towardsdatascience.com/image-filters-in-python-26ee938e57d2)
  

  
  
  
