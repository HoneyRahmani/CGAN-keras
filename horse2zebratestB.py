# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 11:59:27 2020

@author: asus
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
import matplotlib.pyplot as plt

def load_image(filename, size=(256,256)):
    
    pixels = load_img(filename, target_size=(256,256))
    pixels = img_to_array(pixels)
    pixels = expand_dims(pixels,0)
    pixels = (pixels - 127.5)/127.5
    
    return pixels

imge = load_image('C:/Users/asus/Desktop/CGAN/HorsetoZebra/horse2zebra/trainB/n02391049_4.jpg')
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_BtoA_011870.h5', cust)

image_tar = model_AtoB.predict(imge)
image_tar = (image_tar+1)/2.0
imge = (imge+1)/2.0
plt.imshow(imge[0])
plt.show()
plt.imshow(image_tar[0])
plt.show()

