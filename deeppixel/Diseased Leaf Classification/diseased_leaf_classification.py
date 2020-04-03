from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import LabelBinarizer 
import numpy as np 
import mahotas 
import cv2 
import os 
import h5py
import numpy as np 
import os 
import glob 
import cv2 
from matplotlib import pyplot 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.model_selection import KFold, StratifiedKFold 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.externals import joblib

# fixed-sizes for image 
fixed_size = tuple((500, 500))

# no.of.trees for Random Forests 
num_trees = 100

# bins for histogram 
bins = 8

# train_test_split size 
test_size = 0.10

# seed for reproducing same results 
seed = 9

# feature-descriptor-1: Hu Moments 
def fd_hu_moments(image):     
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
  feature = cv2.HuMoments(cv2.moments(image)).flatten()     
  return feature

# feature-descriptor-2: Haralick Texture 
def fd_haralick(image):     
  # convert the image to grayscale     
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
  # compute the haralick texture feature vector     
  haralick = mahotas.features.haralick(gray).mean(axis=0)     
  # return the result     
  return haralick

# feature-descriptor-3: Color Histogram 
def fd_histogram(image, mask=None):     
  # convert the image to HSV color-space     
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)     
  # compute the color histogram     
  hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256]) 
  # normalize the histogram     
  cv2.normalize(hist, hist)     
  # return the histogram     
  return hist.flatten()

# path to training data 
train_path = "/content/" 
# get the training labels 
train_labels = os.listdir(train_path) 
# sort the training labels 
train_labels.sort() 
print(train_labels) 
# empty lists to hold feature vectors and labels 
global_features = [] 
labels = [] 
i, j = 0, 0 
k = 0 
# num of images per class 
images_per_class = 104

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in os.listdir(dir):

        # read the image and resize it to a fixed-size
        image = cv2.imread(os.path.join(dir, x))
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        labels.append(current_label)
        global_features.append(global_feature)


# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

# save the feature vector using HDF5
h5f_data = h5py.File('/content/drive/My Drive/Hdata.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('/content/drive/My Drive/Hlabels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

models = [] 
models.append(('LR', LogisticRegression(random_state=9))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier(random_state=9))) 
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9))) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC(random_state=9))) 

# variables to hold the results and names 
results = [] 
names = [] 
scoring = "accuracy" 

# import the feature vector and trained labels 
h5f_data = h5py.File('/content/drive/My Drive/Hdata.h5', 'r') 
h5f_label = h5py.File('/content/drive/My Drive/Hlabels.h5', 'r') 
global_features_string = h5f_data['dataset_1'] 
global_labels_string = h5f_label['dataset_1'] 
global_features = np.array(global_features_string) 
global_labels = np.array(global_labels_string) 
h5f_data.close() 
h5f_label.close() 


# split the training and testing data 
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),np.array(global_labels),test_size=test_size,random_state=seed) 

for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# create the model - Random Forests 
clf  = RandomForestClassifier(n_estimators=100, random_state=9) 
# fit the training data to the model 
clf.fit(trainDataGlobal, trainLabelsGlobal) 

# path to test data 
test_path = "/content/drive/" 

# loop through the test images 
for x in os.listdir(test_path):     
  image = cv2.imread(os.path.join(test_path, x))      
  image = cv2.resize(image, fixed_size)         
  fv_hu_moments = fd_hu_moments(image)     
  fv_haralick   = fd_haralick(image)     
  fv_histogram  = fd_histogram(image)        
  global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])   
  
  #predict whether the leaf is diseased or healthy
  prediction = clf.predict(global_feature.reshape(1,-1))[0]     
  cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)        
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))     
  plt.show()