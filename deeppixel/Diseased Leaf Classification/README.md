# Diseased Leaf Classification Using Global Feature Extraction

## Aim:
Crop diseases are a major threat to food security, but their rapid identification remains difficult in many parts of the world due to the lack of the necessary infrastructure. 
The combination of increasing global smartphone penetration and recent advances in computer vision made possible by deep learning has paved the way for smartphone-assisted disease diagnosis.
This project compares the performance of different classifiers and the classifier with highest accuracy is used to predict the diseased and healthy leaf.

## DataSet Used:
PlantVillage Dataset from Kaggle is used which includes 22 types of different plants bot diseased and non diseased each comprising 
of 104 images which means total 2288 images, out of which 80% are used for training and 20% are used for testing.

Link - https://www.kaggle.com/emmarex/plantdisease

## Approach:
1.) Loading and resizing the dataset.

2.) Extracting various features using Global Feature Extraction techniques such as extracting shape using 
    Hu Moments, extracting color using Color of Histogram and extracting texture using Haralick Texture.
    
3.) Traing the model using various classifiers such as LinearDiscriminantAnalysis, KNeighborsClassifier, DecisionTreeClassifier,
    RandomForestClassifier, GaussianNB and SVM.
    
4.) Comparing their accuracy using Box Plot visualization and using the classifier with highest accuracy to train and test the model.

5.) As Random Forest clasifier classified with highest accuracy, we finally trained our model using Random Forest Classifier.

### Input Image (Grape Esca leaf)<br><br>

![Input](https://github.com/kritika12298/DeepPixel/blob/master/deeppixel/Diseased%20Leaf%20Classification/Input/t_1.JPG?raw=true)

### Output Image <br><br>
![Output](https://github.com/kritika12298/DeepPixel/blob/master/deeppixel/Diseased%20Leaf%20Classification/Output/GrapeEsca.png?raw=true)

## Paper Implemented:

https://ieeexplore.ieee.org/document/8437085
