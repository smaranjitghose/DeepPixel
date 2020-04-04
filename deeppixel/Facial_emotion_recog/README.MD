# Facial Expression Recogintion

 Facial expressions are one of the most important mediums of communication in human beings. It has many applications such as driver saftey, cognitive science, healthcare and virtual reality. It is the task of classifying faces into different categories based on emotions such as happy, sad, angry, fear, disgust and many more. The process majorly consists of preprocessing, feature extraction, feature selection and then classification. Here, we have used Convolutional Neural Networks to solve this problem. In this basic approach, we have used FER2013 dataset as our training data. After the model is saved, we load the model and display the results on few handpicked images from the JAFFE and CASIA WebFace dataset.
 
### Further Work Planned
1. The current accuracy stands at 90% and the model is not robust. Improve the accuracy by changing the model architecture. 
2. Currently, Input images are provided in the google colab. Working on the task to provide script to feed in input images to the pre-trained model directly. 
3. Working on introducing web-cam as a source of input. 

### Reference 

1. [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. [JAFFE](https://zenodo.org/record/3451524#.XojAEYgzbIU)
3. [CASIA WebFace](https://pgram.com/dataset/casia-webface/)
4. [Facial_Expression_Recognition_Using_Convolutional_Neural_Network](https://www.researchgate.net/publication/328817338_Facial_Expression_Recognition_Using_Convolutional_Neural_Network)