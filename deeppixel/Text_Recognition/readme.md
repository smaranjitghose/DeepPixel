# Text Recognition

## Aim:
The aim of the model is to identify the text from a given image. It performs Optical Character Recognition (OCR) using Google's Tesseract engine



## Approach:
1. Load the example image and convert it to grayscale
2. Apply the thresholding on the image after checking the value of proprocess. The value can either be thresh or blur 
3. Write the grayscale image to disk as a temporary file
4. Load the image as a Pillow image
5. Apply Optical Character Recognition 
6. Display the text and the resulting images


### Required libraries

1. pytesseract
2. numpy 
3. opencv
4. argparse
5. PIL


## Reference:

[OpenCV documentation](https://docs.opencv.org/3.2.0/db/d7b/group__datasets__tr.html) <br/>
[Tesseract ](https://github.com/tesseract-ocr/tesseract)


