# Colorization of black-and-white images
 
Image colorization is the process of taking an input grayscale (black and white) image and then producing an output colorized image  
 
![colorization-example](https://user-images.githubusercontent.com/43414928/77244083-67481600-6c37-11ea-9903-b68c059bbbec.png)
 
 
## How does it work? 
The approach we will follow is :
1. Convert the RGB color image into the Lab color space. Similar to the RGB color space, the Lab color space has three channels:-  
   * 'L' channel denotes lightness (intensity information),
   * 'a' is the green-red component 
   * 'b' is the blue-yellow component. 
2. The input image is a grayscale image, which can be considered as the 'L' component of the Lab color space.
3. We feed this 'L' component into our trained model, the model then predicts the 'a' and 'b' components. 
4. We then concatenate the L,a,b  components to generate Lab image. 
5. This Lab image is then converted to rgb image for better visualization
![function_lab_color_grids](https://user-images.githubusercontent.com/43414928/77244456-e723af80-6c3a-11ea-8a09-ea4d35f20f6e.png)
 
 
## Required libraries
Tensorflow 
NumPy
matplotlib
OpenCV
argparse
 
## Steps 
 
1. Clone this repository:
```
git clone https://github.com/smaranjitghose/DeepPixel
```
2. Navigate to the project directory bw_to_c
 
2. Install the dependencies:
```
pip install -r requirements.txt 
```
 
3. Download the **colorization_release_v2.caffemodel** from https://github.com/richzhang/colorization/blob/master/models/fetch_release_models.sh                             
   and move them to **model** folder.
 
4. Run **colorization.py** file with command: 
 
```
python colorization.py [INPUT_PATH] -f [FORMAT] -o [OUTPUT_PATH]
```
 > For example
```
python colorization.py input/ -f jpg -o output/
```
 
 
6. This will save the colorized images in the output folder path provided.
 
 
 
## Reference 

1. [Colorful Image Colorization - Richard Zhang ](https://richzhang.github.io/colorization/resources/colorful_eccv2016.pdf)
2. [Black and white image colorization ](https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning)

