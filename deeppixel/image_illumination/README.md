# Image Ilumination

The aim of this task is to provide image illumination. Such image enhancement can be usually done by providing 
alpha and beta values as input and then applying gamma correction methology. In our code, we provide adaptive 
image illumination by automatic generation of alpha and beta values and then applying gamma correction. 


## Approach :
The approach we will follow is :
1.  First we take a low illumination image, and threshold value as input (Here we have taken threshold as 24)
2.  Now we convert the input image to gray scale.
3. After converting it to gray scale, we calculate the histogram of the image values.
3. Cummulative distribution of the histogram is found out, to determine the color frequency which might be less than the threshold.
4. The histogram is clipped by removing the part to the left and right side of the threshold. Thus, we also get the maimum and minum range values. 
5. After we get the minimum and maximum range, we calculate the alpha and beta values.
6. These values are now used, to find the enhanced image. 
