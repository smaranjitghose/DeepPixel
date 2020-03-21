# Colorization of black-and-white images

The aim of this task is to generate a script which produces **colorized images**, when **black and white images** are provided as input. 

The script is written in the colorization.py file. For in-depth understanding of the code you may refer to the image_colorization.ipynb file.

## Approach :
The approach we will follow is :
1. For this task we use Lab color space in which 'L' channel denotes lightness (intensity information),
   'a' is the green-red component and 'b' is the blue-yellow component. 
2. The input image is a grayscale image, which can be considered as the 'L' component of the Lab color space.
3. We feed this 'L' component into our trained model, the model then predicts the 'a' and 'b' components. 
4. We then concatenate the L,a,b  components to generate Lab image. 
5. This Lab image is then converted to rgb image for better visualization

## Steps :

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
python colorization.py [INPUT_PATH] -f [INPUT_FORMAT] -o [OUTPUT_PATH]
```
 > for example
```
python colorization.py input/ -f jpg -o output/
```


6. This will save the colorized images in the output folder path provided.
