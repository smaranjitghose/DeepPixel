# Image Denoising

## Aim:
To extract a clear image from their low light images.

## DataSet Used:
Flowers Recognition DaataSet is used from Kaggle which comprises 4319 images of 6 different flower species. 
Then the images are converted into their low light images such that the original and low light images are used
for training.

Link: https://www.kaggle.com/alxmamaev/flowers-recognition

## Approach:
1.) Loading and resizing the dataset.
2.) Creating image arrays.
3.) Creating the UNet Model for training using keras.
4.) Saving the weights and predicting the output for test images.

## Paper Implemented:
http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1981.pdf

## Colab Link:
https://colab.research.google.com/drive/1qB3-GhtudWHzKbhO2AIOqXhudZVbsLXO

## References:
https://github.com/cchen156/Learning-to-See-in-the-Dark
https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn
