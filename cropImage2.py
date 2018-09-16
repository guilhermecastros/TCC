# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 09:10:20 2018

@author: Guilherme
"""

from PIL import Image
import os
import random
import cv2
from random import shuffle
import glob

def crop(infile,folderName,height,width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    count = 0
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            cropped_image = im.crop(box)
            cropped_image.save('Exemplo 3/' + folderName + '/image' + str(count) + '.jpg')
            count = count + 1


#--------------- Criando a máscara de binária das estradas ---------------#
image = Image.open('Exemplo 3/Mapa_com_estradas.jpg')
pix = image.load()
count = 0
width, height = image.size
for w in range(width):
    for h in range(height):
        r,g,b = pix[(w,h)]
        if (r > 150 and g > 150 and b < 50):
            pix[(w,h)] = 255, 0, 0
            print('r: ' + str(r) + ', ' + 'g: ' + str(g) + ', ' + 'b: ' + str(b))
            #count = count + 1
            #if (count < 30):
            pix[(w,h)] = 255, 255, 255
        else:
            pix[(w,h)] = 0, 0, 0

image.save('Exemplo 3/Mapa_estradas.jpg')

#--------------- Crop Imagem ---------------#
count = 0
width = 64
height = 64
start_num = 1

imageRoads = 'Exemplo 3/Mapa_estradas.jpg'
folderName = 'cropImgRoads'
crop(imageRoads,folderName,height,width)

imageOriginal = 'Exemplo 3/Mapa_original.jpg'
folderName = 'cropImgOriginal'
crop(imageOriginal,folderName,height,width)


#--------------- Separar imagens com estradas ---------------#
listOfImages = [f for f in os.listdir('Exemplo 3/cropImgRoads') if os.path.isfile(os.path.join('Exemplo 3/cropImgRoads', f))]

for imageName in listOfImages:
    roadImg = Image.open('Exemplo 3/cropImgRoads/' + imageName)
    originalImg = Image.open('Exemplo 3/cropImgOriginal/' + imageName)
    
    width, height = roadImg.size
    pix = roadImg.load()
    count = 0
    for w in range(width):
        for h in range(height):
            r,g,b = pix[(w,h)]
            if (r == 255 and g == 255 and b == 255):
                count = count + 1
                
    if (count > 20):
        originalImg.save('Exemplo 3/roads/' + imageName)
    else:
        originalImg.save('Exemplo 3/earth/' + imageName)
        

#--------------- Selecionando imagens sem estradas ---------------#
listOfEarthImages = [f for f in os.listdir('Exemplo 3/earth') if os.path.isfile(os.path.join('Exemplo 3/earth', f))]
listOfRoadImages = [f for f in os.listdir('Exemplo 3/roads') if os.path.isfile(os.path.join('Exemplo 3/roads', f))]

selectedEarthImgs = random.sample(range(1, len(listOfImages)), len(listOfRoadImages))

for imageIndex in selectedEarthImgs:
    earthImg = Image.open('Exemplo 3/earth/' + listOfEarthImages[imageIndex])
    earthImg.save('Exemplo 3/selectedEarth/' + listOfEarthImages[imageIndex])
    
    
    
#--------------- Criando o label dos dados ---------------#

listOfSelectedEarthImages = [f for f in os.listdir('Exemplo 3/selectedEarth') if os.path.isfile(os.path.join('Exemplo 3/selectedEarth', f))]
listOfRoadImages = [f for f in os.listdir('Exemplo 3/roads') if os.path.isfile(os.path.join('Exemplo 3/roads', f))]
    
labelList = []
for image in listOfSelectedEarthImages:
    #imagePath = glob.glob('Exemplo 3/selectedEarth/' + image)[0]
    im = cv2.imread('Exemplo 3/selectedEarth/' + image)
    #Image.open('Exemplo 3/selectedEarth/' + image)
    newImage = {'name': image, 'label': 'earth', 'data': im }
    labelList.append(newImage)
    
for image in listOfRoadImages:
    im = cv2.imread('Exemplo 3/roads/' + image)
    newImage = {'name': image, 'label': 'road', 'data': im }
    labelList.append(newImage)
    
shuffle(labelList)

print(labelList[2]['label'])
labelList[2]['data']

#------------ Utilizando a biblioteca Keras para classificação ------------#

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
print('Original label 0 : ', y_train[0])
print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

def createModel():
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))
    
    return model


model1 = createModel()
batch_size = 256
epochs = 50
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model1.summary()

history = model1.fit(X_train, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(X_test, test_labels_one_hot))

model1.evaluate(X_test, test_labels_one_hot)


