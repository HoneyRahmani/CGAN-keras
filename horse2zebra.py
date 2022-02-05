# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:03:46 2020

@author: asus
"""
import numpy as np
from skimage.transform import resize
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import matplotlib.pyplot as plt
#from Resize_Image import Resize_Image
'''
from os import listdir
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed


def load_image(path, size=(256,256)):
    
    data_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        data_list.append(pixels)
    return asarray(data_list)

#loading all images into memory
path = 'C:/Users/asus/Desktop/CGAN/HorsetoZebra/horse2zebra/'

dataA1 = load_image(path + 'trainA/')
dataAB = load_image(path + 'testA/')
dataA = vstack((dataA1,dataAB))


dataB1 = load_image(path + 'trainB/')
dataBA = load_image(path + 'testB/')
dataB = vstack((dataB1,dataBA))

filename = 'horse2zebra_256.npz'
savez_compressed(filename,dataA,dataB)
print("Saved dataset:", filename)
'''

def define_discriminator(image_shape):
    
    init = RandomNormal(stddev = 0.02)
    
    in_image = Input(shape = image_shape)
    
    # C64
    d = Conv2D(16, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # 
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    
    return model


#model = define_discriminator(image_shape)
#model.summary()
#print(model.output_shape[1])
#plot_model(model, to_file='discriminator_model_plot.png', show_shapes=True,show_layer_names=True)

def resnet_block (n_filters, input_layer):
    
    init = RandomNormal(stddev=0.02)
    
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    
    g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    
    g = concatenate()([g, input_layer])
    
    return g

def define_generator (image_shape, n_resnet=6):
    
    init = RandomNormal(stddev=0.02)
    
    in_image = Input(shape=image_shape)
    
    g = Conv2D(16, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    
    g = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    
    g = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g=Activation('relu')(g)
    
    for _ in range(n_resnet): 
        g = resnet_block(64,g)
        
    g = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    
    g = Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    
    g = Conv2D(3, (7,7),padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    
    model = Model(in_image,out_image)
    
    return model


#model = define_generator()
#model.summary()   
#plot_model(model, to_file='discriminator_model_plot.png', show_shapes=True,show_layer_names=True)    



def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    #The weights of the other models are marked as not trainable 
    #as we are only interested in updating the first generator model
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False
    
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    #The discriminator is connected to the output of the generator in order to classify generated
    #images as real or fake
    output_d = d_model(gen1_out)
    #identity mapping,g_model_1 generates the target domain 
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    #forward cycle
    output_f = g_model_2(gen1_out)
    #the backward cycle
    gen2_output = g_model_2(input_id)
    output_b = g_model_1(gen2_output)
    
    model = Model([input_gen,input_id],[output_d,output_id,output_f,output_b])
    
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1,5,10,10], 
                  optimizer=opt)
    return model



def generate_real_samples(dataset, n_samples, patch_shape):
    
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    
    return X, y

def generate_fake_samples(g_model, dataset, patch_shape):
    
    X = g_model.predict(dataset)
    y = zeros((len(X), patch_shape, patch_shape,1))
    
    return X,y

def update_image_pool(pool, images, max_size=50):
    
    selected = list()
    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            ix = randint(0,len(pool))
            selected.append(pool[ix])
            pool[ix]=image  
    return asarray(selected)
def Resize_Image (Img,newheight, newweight, newchanel):
       
# 1- Resize image
    
    Img_re = np.zeros((Img.shape[0],newheight, newweight, newchanel),
                     dtype=np.float32)
    
    for i in range(Img.shape[0]):
        
        Img_re[i] = resize(Img[i],(newheight, newweight, newchanel),
                    preserve_range = True,
                    mode='constant',
                    anti_aliasing=True)
    return Img_re
# load and prepare training images
def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    newheight=128
    newweight=128 
    newchanel=3
    X11 = Resize_Image(X1,newheight, newweight, newchanel)
    X22 = Resize_Image(X2,newheight, newweight, newchanel)
    return [X11, X22]  

def save_models(step, g_model_AtoB, g_model_BtoA):
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
    
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=5):
    # select a sample of input images
    X_in, _ = generate_real_samples(trainX, n_samples, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_in[i])
    # plot translated image
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(X_out[i])
     # save plot to file
    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))
    plt.savefig(filename1)
    plt.close()
   
  

def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, 
          dataset):
    
    n_epochs, n_batch = 20,1
    n_patch = d_model_A.output_shape[1]
    
    
    trainA, trainB = dataset
    
    poolA, poolB = list(), list()
    bat_per_epo = int((len(trainA) / n_batch))
    
    
    n_steps = bat_per_epo * n_epochs
    for i in range (n_steps):
        
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        
        gloss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], 
                                                         [y_realA, X_realA, X_realB, X_realA])
        
        
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
       
        
        
        gloss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], 
                                                         [y_realB, X_realB, X_realA, X_realB])
        

        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        

        
        print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2,
                      dB_loss1,dB_loss2, gloss1, gloss2))
        if (i+1) % (bat_per_epo * 1) == 0:
            # plot A->B translation
            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            # plot B->A translation
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
        if (i+1) % (bat_per_epo * 5) == 0:
            # save the models
            save_models(i, g_model_AtoB, g_model_BtoA)
            
        

      
# load image data
dataset = load_real_samples('horse2zebra_256.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]

    

# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)

# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)    
