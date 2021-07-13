import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

import skimage.io
from skimage.transform import resize

### Here we load in all the relevant neural network packages ###
import tensorflow as tf

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

def load_image(path):
    """Load an image file from path."""
    img = skimage.io.imread(path, as_grey=True)
    return img

def load_mask(path):
    """Load a mask file from path, scaling to 0/1."""
    mask = skimage.io.imread(path, as_grey=True)
    mask = mask / 65535
    return mask

def biggenate(x, target_image_size = (128,128,1)):
    """Resize an image or mask, making it bigger based on target_image_size"""
    return resize(x, target_image_size, mode='constant', preserve_range=True)

def smallenate(x, target_image_size = (101,101)):
    """Resize an image or mask, making it smaller based on target_image_size"""
    return resize(x, target_image_size, mode='constant', preserve_range=True)

def biggenate_zero_pad(x, target_image_size = (128,128,1)):
    """Resize an image or mask, making it bigger based on target_image_size by zero-padding to this size"""
    new_image = np.zeros((target_image_size[0], target_image_size[1], target_image_size[2]), dtype=np.float64)
    new_image[13:114,13:114, 0] = x
    return new_image

def smallenate_zero_pad(x):
    """Resize an image or mask, cropping it to the required size."""
    new_image = x[13:114,13:114]
    return new_image

class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """

        pred = []
        for x_i in X:
            tmp = x_i
            p0 = self.model.predict(tmp.reshape(1,128,128,1))
            p1 = self.model.predict(np.fliplr(tmp).reshape(1,128,128,1))
#             p2 = self.model.predict(np.flipud(tmp).reshape(1,128,128,1))
#             p3 = self.model.predict(np.fliplr(np.flipud(tmp)).reshape(1,128,128,1))
            p = (p0[0] +
                 np.fliplr(p1[0]) #+
#                  np.flipud(p2[0]) +
#                  np.fliplr(np.flipud(p3[0]))
                 ) / 2#4
            pred.append(p)
        return np.array(pred)
    
class TTA_ModelWrapper_3channel():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """

        pred = []
        for x_i in X:
            tmp = x_i
            p0 = self.model.predict(tmp.reshape(1,128,128,3))
            p1 = self.model.predict(np.fliplr(tmp).reshape(1,128,128,3))
#             p2 = self.model.predict(np.flipud(tmp).reshape(1,128,128,1))
#             p3 = self.model.predict(np.fliplr(np.flipud(tmp)).reshape(1,128,128,1))
            p = (p0[0] +
                 np.fliplr(p1[0]) #+
#                  np.flipud(p2[0]) +
#                  np.fliplr(np.flipud(p3[0]))
                 ) / 2#4
            pred.append(p)
        return np.array(pred)

def mask_to_rle(mask):
    """Takes a mask that is nxm, represented as 0 and integers, or as Boolean,
    and converts it to run length encoding."""
    mask = np.transpose(mask)
    mask = mask.astype(np.bool).astype('int')
    mask = mask.reshape(-1)

    dummy = np.insert(mask, 0, 0)
    mask = np.insert(mask, len(mask), 0)

    diff = dummy - mask
    
    diffindx = np.argwhere(diff!=0)
    
    starts = diffindx[::2] + 1 # it is one indexed so need this
    
    runs = diffindx[1::2] - diffindx[::2]
    
    run_length_encoded = np.hstack((starts, runs)).reshape(-1)
    
    return run_length_encoded