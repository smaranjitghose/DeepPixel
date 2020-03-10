The aim of this code is colorization of black-and-white images
Input : Black and white image, Expected Output : Colorized image
Process : 
1. For this task we use Lab color space in which 'L' channel denotes lightness (intensity information),
   'a' is the green-red component and 'b' is the blue-yellow component. 
2. The input image is a grayscale image, which can be considered as the 'L' component of the Lab color space.
3. We feed this 'L' component into our trained model, the model then predicts the 'a' and 'b' components. 
4. We then concatenate the L,a,b  components to generate Lab image. 
5. This Lab image is then converted to rgb image for better visualization
