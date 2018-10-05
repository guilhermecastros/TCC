# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:34:44 2018

@author: Guilherme
"""

from PIL import Image
import os
import cv2
from random import shuffle



#--------------- Criando o label dos dados ---------------#
listOfSelectedEarthImages = [f for f in os.listdir('Exemplo 4/selectedEarth') if os.path.isfile(os.path.join('Exemplo 4/earth', f))]
listOfRoadImages = [f for f in os.listdir('Exemplo 4/roads') if os.path.isfile(os.path.join('Exemplo 4/roads', f))]
    
labelList = []
for image in listOfSelectedEarthImages:
    #imagePath = glob.glob('Exemplo 3/selectedEarth/' + image)[0]
    im = cv2.imread('Exemplo 4/earth/' + image)
    #Image.open('Exemplo 3/selectedEarth/' + image)
    newImage = {'name': image, 'label': 'earth', 'data': im }
    labelList.append(newImage)
    
for image in listOfRoadImages:
    im = cv2.imread('Exemplo 4/roads/' + image)
    newImage = {'name': image, 'label': 'road', 'data': im }
    labelList.append(newImage)
    
shuffle(labelList)


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