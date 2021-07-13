import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

import skimage.io
from skimage.transform import resize

### Here we load in all the relevant neural network packages ###
import tensorflow as tf

from keras.applications import vgg16, vgg19, mobilenet

from keras.models import Model
from keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, concatenate, Conv2DTranspose, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
# from keras import applications
from keras.preprocessing.image import ImageDataGenerator

def unet_model(filter_scaling=16, depth=5, batch_norm_momentum=0.6, optimizer='adam'):
    """Function to produce untrained U-Net models mostly following the original U-net paper.
    It is designed so that the U-Net can be scaled automatically to be deeper/shallower during
    model initialization. Additionally, the number of convolutional filters in each layer
    can be adjusted during initialization.

    filter_scaling: Different layers of the U-Net will have multiples of the previous layer
    in terms of the number of convoluational filters. This allows for the base rate of filters
    in the initial layer, from which all others are derived, to be increased/decreased.

    depth: The number of times the ladder repeats during its descent before repeating back up.

    batch_norm_momentum: The normal momentum hyperparameter for all the BatchNorm layers in the U-Net

    optimizer: Optimizer used to make the actual model weight/bias updates based on the gradients.
    """
    input_layer = Input((128,128,1))

    conv_initialization_dict = {"activation":'relu', 
                                "padding":'same',
                                "kernel_initializer" : 'he_normal'}

    conv_initialization_dict_no_activation = {"padding":'same',
                                "kernel_initializer" : 'he_normal'}

    conv_dict = {}
    for i in range(1,depth):
        if i==1:
            x = Conv2D((2**(i-1)) * filter_scaling, (3,3), **conv_initialization_dict_no_activation)(input_layer)
        else:
            x = Conv2D((2**(i-1)) * filter_scaling, (3,3), **conv_initialization_dict_no_activation)(x)
        x = Activation('relu')(x)     
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        conv_dict[i] = Conv2D((2**(i-1)) * filter_scaling, (3,3), **conv_initialization_dict_no_activation)(x)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)

        x = MaxPooling2D((2,2), padding='same')(conv_dict[i])


    ### The bottom of the network ###
    x = Conv2D((2**(depth-1)) * filter_scaling, (3,3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D((2**(depth-1)) * filter_scaling, (3,3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)

    for i in range(depth-1,0,-1):

        x = Conv2DTranspose((2**(i-1)) * filter_scaling, (3,3), strides=(2,2), **conv_initialization_dict)(x)
        x = concatenate([conv_dict[i], x])
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = Conv2D((2**(i-1)) * filter_scaling, (3,3), **conv_initialization_dict)(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = Conv2D((2**(i-1)) * filter_scaling, (3,3), **conv_initialization_dict)(x)
        if i!=1:
            x = BatchNormalization(momentum=batch_norm_momentum)(x)

    output_layer = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)

    model = Model(input_layer, output_layer)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model

def transfer_unet_model(freeze_layer=11, batch_norm_momentum=0.7):
    """U-Net model that uses a pre-trained VGG-16 as the encoder side of the ladder. The decoder
    part of the ladder that produces the final mask is initialized with the He normal initialization
    with the matching VGG16 geometry. Using the pre-trained model allows us to get the benefit of
    transfer learning of the features that VGG-16 learned during its original ImageNet training.
    
    freeze_layer: We freeze some of the lower level (closer to the input image) features in 
    the pre-trained VGG-16 model, as is typical during transfer learning. This parameter allows
    us to select where we want this freeze to begin.
    
    batch_norm_momentum: The normal momentum hyperparameter for all the BatchNorm layers in the U-Net"""
    
    base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    ### Freeze some layers, don't want to lose that sweet sweet training ###
    for layer in base_model.layers[:freeze_layer]:
        layer.trainable = False

    for layer in base_model.layers[freeze_layer:]:
        layer.trainable = True

    conv_initialization_dict = {"activation":'relu', 
                                    "padding":'same',
                                    "kernel_initializer" : 'he_normal'}

    ### Get the last convolutional layer, we will call that layer the bottom of the ladder and start climbing again ###
    x = base_model.get_layer('block4_pool').output
    # x = base_model.get_layer('block5_conv3').output

    ### Added this as the new bottom layer ###
    x = Conv2D(1024, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(1024, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)

    x = Conv2DTranspose(512, (3, 3), strides=(2, 2), **conv_initialization_dict)(x)
    x = concatenate([x, base_model.get_layer('block4_conv3').output])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(512, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(512, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)

    x = Conv2DTranspose(256, (3, 3), strides=(2, 2), **conv_initialization_dict)(x)
    x = concatenate([x, base_model.get_layer('block3_conv3').output])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(256, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(256, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), **conv_initialization_dict)(x)
    x = concatenate([x, base_model.get_layer('block2_conv2').output])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(128, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(128, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), **conv_initialization_dict)(x)
    x = concatenate([x, base_model.get_layer('block1_conv2').output])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(64, (3, 3), **conv_initialization_dict)(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(64, (3, 3), **conv_initialization_dict)(x)

    output_layer = Conv2D(1, (1,1), activation='sigmoid', padding='same')(x)

    model = Model(base_model.input, output_layer)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model