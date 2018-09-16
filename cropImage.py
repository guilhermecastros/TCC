# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 09:09:18 2018

@author: Guilherme
"""

from PIL import Image
import numpy as np

savedImageList = []
# left < x < right and upper < y < lower
def checkOverlap(newPixel, savedImageList):
    if len(savedImageList) > 0:
        for image in savedImageList:
            if (image['left'] < newPixel[0] < image['right'] and image['upper'] < newPixel[1] < image['lower']):
                return 1
        
        newBox = {'left': newPixel[0] - 32, 'upper': newPixel[1] - 32, 'right': newPixel[0] + 32, 'lower': newPixel[1] + 32}
        savedImageList.append(newBox)
        return 0
    else:
        newBox = {'left': newPixel[0] - 32, 'upper': newPixel[1] - 32, 'right': newPixel[0] + 32, 'lower': newPixel[1] + 32}
        savedImageList.append(newBox)
        return 0


#--------------- Criando a máscara de binária das estradas ---------------#
image = Image.open('Mapa_com_linhas.jpg')
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

image.save('Mapa_estradas.jpg')


image = Image.open('Exemplo 1/Mapa_estradas.jpg')
image2 = Image.open('Exemplo 1/Mapa_original.jpg')
# image.show()
pix = image.load()
count = 0
width, height = image.size
for w in range(width):
    for h in range(height):
        r,g,b = pix[(w,h)]
        if (r == 255 and g == 255 and b == 255):
            left = w - 32
            upper = h - 32
            right = w + 32
            lower = h + 32
            if ((left > 0) and (upper > 0) and (right < width) and (lower < height)):
                box = {'left': left, 'upper': upper, 'right': right, 'lower': lower}
                box = (left, upper, right, lower)
                if (checkOverlap((w,h), savedImageList) == 0):
                    cropped_image = image2.crop((left, upper, right, lower))
                    cropped_image.save('Exemplo 1/cropImg/image' + str(count) + '.jpg')
                    print('count: ' + str(count))
                    #print('height: ' + str(h) + ', ')
                    #print('left: ' + str(left) + ', ')
                    #print('upper: ' + str(upper) + ', ')
                    #print('right: ' + str(right) + ', ')
                    #print('lower: ' + str(lower))
                    #print('----------------------------')
                    count = count + 1