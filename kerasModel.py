# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:10:03 2018

@author: Guilherme
"""

from PIL import Image
import os
import cv2
from random import shuffle

#--------------- Criando o label dos dados ---------------#

listOfSelectedEarthImages = [f for f in os.listdir('Exemplo 4/selectedEarth') if os.path.isfile(os.path.join('Exemplo 4/selectedEarth', f))]
listOfRoadImages = [f for f in os.listdir('Exemplo 4/roads') if os.path.isfile(os.path.join('Exemplo 4/roads', f))]
    
labelList = []
for image in listOfSelectedEarthImages:
    #imagePath = glob.glob('Exemplo 3/selectedEarth/' + image)[0]
    im = cv2.imread('Exemplo 4/selectedEarth/' + image)
    #Image.open('Exemplo 3/selectedEarth/' + image)
    newImage = {'name': image, 'label': 'earth', 'data': im }
    labelList.append(newImage)
    
for image in listOfRoadImages:
    im = cv2.imread('Exemplo 4/roads/' + image)
    newImage = {'name': image, 'label': 'road', 'data': im }
    labelList.append(newImage)
    
shuffle(labelList)

import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical

classes = np.unique([d['label'] for d in labelList if 'label' in d])
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[4,2])

# Display the first image in training data
plt.subplot(121)
plt.imshow(labelList[0]['data'])
plt.title("Image from : {}".format(labelList[0]['label']))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(labelList[1]['data'])
plt.title("Image from : {}".format(labelList[1]['label']))

#--------------- Dividindo os dados ---------------#
from sklearn.model_selection import train_test_split

data = [d['data'] for d in labelList if 'data' in d]
target = [d['label'] for d in labelList if 'label' in d]
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

for index, image in enumerate(X_train):
    if(y_train[index] == 'earth'):
        cv2.imwrite("./Exemplo 4/train/earth/" + str(index) + ".jpg", image)
    else:
        cv2.imwrite("./Exemplo 4/train/road/" + str(index) + ".jpg", image)
        

for index, image in enumerate(X_test):
    if(y_test[index] == 'earth'):
        cv2.imwrite("./Exemplo 4/test/earth/" + str(index) + ".jpg", image)
    else:
        cv2.imwrite("./Exemplo 4/test/road/" + str(index) + ".jpg", image)
    
# Find the shape of input images and create the variable input_shape
nRows,nCols,nDims = X_train[0].shape
# X_train = X_train.reshape(X_train.shape[0], nRows, nCols, nDims)
# X_test = X_test.reshape(X_test.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

# Change to float datatype
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

# Scale the data to lie between 0 to 1
X_train /= 255
X_test /= 255

# Change the labels from integer to categorical data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

train_labels_one_hot = to_categorical(y_train)
test_labels_one_hot = to_categorical(y_test)

# Display the change for category label using one-hot encoding
print('Original label: ', y_train[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

from  keras.applications import VGG16
#Load the VGG model
image_size = 64
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False
 
# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)
        
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# Create the model
model = models.Sequential()
 
# Add the vgg convolutional base model
model.add(vgg_conv)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()

batch_size = 32
epochs = 50
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

datagen = ImageDataGenerator(
#         zoom_range=0.2, # randomly zoom into images
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


history = model.fit_generator(datagen.flow(X_train, train_labels_one_hot, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
                              epochs=epochs,
                              validation_data=(X_test, test_labels_one_hot),
                              workers=4)



history = model.fit(X_train, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(X_test, test_labels_one_hot))

model.evaluate(X_test, test_labels_one_hot)

	
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

#generator -> keras

