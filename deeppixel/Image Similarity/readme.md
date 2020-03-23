# Image Similarity

 
Two images are said to be completely equal if they have the same size, same channels, and same pixels values. Structural Similarity Index is one method to measure the similarity between two images. Result of comparison between two images using SSIM, gives a value between 0 and 1. Closer to 1 implies more similarity between the images.
 
 **AIM**
 The aim of this task is to take two images as inputs and compare the similarity between them.
 
 
### Approach
The approach we will follow is :
1. First we load the two images that are saved in our directory.
2. We convert these images into gray scale
3. Next we compute the Structural Similarity Index (SSIM) between the two images.
4. If the SSIM = 1, then the two images are identical. Else, we find the difference score between the 2 images 



## Steps

1. Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```
2. Navigate to the project directory Image Similarity
 
3. Install the required libraries
 
4. Run **ImageSimilarity.py** file with command: 
 
```
python ImageSimilarity.py --first [IMAGE1_PATH]  --second [IMAGE2_PATH]
```
 > For example
```
python ImageSimilarity.py --first images/original_02.png --second images/modified_02.png

```
 ![image](https://user-images.githubusercontent.com/43414928/77290112-7398a580-6d01-11ea-8873-ba2307427c24.png)

 
6. After running this command, you would get the similarity score between the two images
 
### Required libraries
1. scikitimage
2. opencv
3. argparse

 
### Reference 

1. [Image Classification using SSIM](https://towardsdatascience.com/image-classification-using-ssim-34e549ec6e12)
2. [Calculate the Structural Similarity Index (SSIM) between two images with Python ](https://ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python)

