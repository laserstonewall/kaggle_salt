import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

import skimage.io
from skimage.transform import resize

### Here we load in all the relevant neural network packages ###
import tensorflow as tf

# from keras.applications import xception, vgg16, vgg19, mobilenet
# from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, concatenate, Conv2DTranspose, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.utils import np_utils, multi_gpu_model
# from keras.optimizers import SGD, Adam
# from keras.datasets import mnist
from keras import backend as K
# from keras import applications
from keras.preprocessing.image import ImageDataGenerator

#######################################################################################
###### Important functions ############################################################
#######################################################################################

def filter_directory_list(directorylisting):
    if '.DS_Store' in directorylisting:
        directorylisting.remove('.DS_Store')
    return directorylisting

def IoU(x,y):
    '''
    version of IoU that uses np.bincount to get the value counts
    
    x and y are both numpy N x M masks
    
    x = proposed mask
    y = ground truth mask
    
    0 for a pixel indicates the mask is blocked, 1 indicates the mask is not blocked.
    In plain English, everywhere that is 1 we can see the cell, everywhere that is 0 we cannot.
    
    We want to calculate the IoU statistic, which is intersection(x,y)/union(x,y) at values where x or y is 1 
    
    By subtracting the proposed mask from 2 x the ground truth mask (i.e. blocked is 0, not blocked is 2),
    then adding 1, we get unique values for each type of overlap situation, plus all values are positive, which
    is required to use np.bincount:
    
INDX  0  1  2  3  4  5  6  7  8  9 10 11

GT    0  0  0  2  2  2  2  2  0  0  0  0
MSK - 0  0  1  1  1  1  0  1  1  0  0  0  
      ----------------------------------
      0  0 -1  1  1  1  2  1 -1  0  0  0
    + 1  1  1  1  1  1  1  1  1  1  1  1
      ----------------------------------
      1  1  0  2  2  2  3  2  0  1  1  1
      
    0: the proposed mask had a pixel, ground truth did not (include in union)   
    1: neither mask had a pixel (don't include)
    2: the proposed mask had a pixed, the ground truth had a pixel (include in intersection and union)
    3: the proposed mask did not have a pixel, the ground truth did (include in union)
    
    np.bincount always has length of np.amax(x) + 1, so we just need to do length checking
    '''
    x = x
    y = y * 2
    
    diff = np.bincount((y - x + 1).flatten())
    diff_len = len(diff)
    
    ### Cacluate the intersection first
    intersection = 0
    if (diff_len >= 3):
        intersection = diff[2]
    
    ### Now calculate the union
    union = intersection
    if diff_len == 4:
        union += diff[3]
    union += diff[0]
        
    if union==0:
        iou = 0 ### default value, we could potentially return blank masks, although GT should never be empty
    else:
        iou = float(intersection) / union

    return iou

def pred_to_binary_mask(pred, threshold):
    tst = np.zeros((pred.shape[0],pred.shape[1]), dtype=np.int8)
    tst[pred >= threshold] = 1
    return tst
    
def calc_iou(pred, gt, threshold):
    pred_mask = pred_to_binary_mask(pred, threshold)
    if (pred_mask.sum()==0) and (gt.sum()==0):
        IoU_value = 1
    elif pred.sum==0 and mask.sum()!=0:
        IoU_value = 0
    else:
        IoU_value = IoU(pred_mask,gt)
    return IoU_value

def calc_avg_precision(pred_mask, gt_mask, threshold, iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    ### First calculate the IoU matrix
    iou = calc_iou(pred_mask, gt_mask, threshold)
    
    avg_precision = (iou_thresholds < iou).sum() / len(iou_thresholds)
        
    return avg_precision

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

def load_image(path):
    img = skimage.io.imread(path, as_grey=True)
    return img

def load_mask(path):
    mask = skimage.io.imread(path, as_grey=True)
    mask = mask / 65535
    return mask

def biggenate(x):
    target_image_size = (128,128,1)
    return resize(x, target_image_size, mode='constant', preserve_range=True)

def smallenate(x):
    target_image_size = (101,101)
    return resize(x, target_image_size, mode='constant', preserve_range=True)

def biggenate_zero_pad(x):
    target_image_size = (128,128,1)
    new_image = np.zeros((target_image_size[0], target_image_size[1], target_image_size[2]), dtype=np.float64)
    new_image[13:114,13:114, 0] = x
    return new_image

def smallenate_zero_pad(x):
    new_image = x[13:114,13:114]
    return new_image

class LearningRateHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
       self.lr.append(K.eval(self.model.optimizer.lr))

##########################################################################################
########### Load in initial images, do some pre-processing ###############################
##########################################################################################

### Load in the whole list then eliminate those that are bad ###
traindir = '../../train/'
all_image_files = np.array(os.listdir(filter_directory_list(traindir + 'images/')))

good_training_images = all_image_files

### Load in the images ###
images_orig = np.array([load_image(traindir + 'images/' + x) for x in good_training_images])
masks_orig = np.array([load_mask(traindir + 'masks/' + x) for x in good_training_images])

### Run this to mean center the images ###
images_orig_m1_to_1 = np.array([x - x.mean() for x in images_orig])

### Run this to use -1 to 1 images ###
images = np.array([biggenate_zero_pad(x) for x in images_orig_m1_to_1])
masks = np.array([biggenate_zero_pad(x) for x in masks_orig])

### Calculate the mask coverage to do stratified sampling ###
### Find the area of the grid ###
area = masks_orig.shape[1] * masks_orig.shape[2]

### Find the fractional covereage of each mask (they are already normalized to [0,1]) ###
coverage = masks_orig.sum(axis=(1,2)) / area

### Cast into a category every 0.1 coverage. 0 coverage gets its own class too ###
coverage_category = np.ceil(coverage * 10)

def unet_model(filter_scaling=16, depth=5, batch_norm_momentum=0.6, optimizer='adam'):

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

# optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
optimizers = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

batch_norm_momentum = 0.6
number_folds = range(1,5)

filter_scaling = 48
depth = 5
batch_size = 64

record_file = "optimizer_comparison_1.csv"
# if os.path.exists(record_file):
#     raise ValueError("Record file already exists")
    
# with(open(record_file,'w')) as f:
#     f.write('filter_scaling,depth,batchsize,fold,bn_momentum,optimizer,final_epoch,highest_val_acc,lowest_val_loss,threshold_max,mAP_max,threshold_max_augmentation,mAP_max_augmentation')

for optimizer in optimizers:
    for fold in number_folds:

        ##########################################################################################
        ###### Create the current train / test setups ###########################################
        ########################################################################################

        train_images, val_images, train_masks, val_masks, train_orig_masks, val_orig_masks, \
        train_image_files, val_image_files, train_indices, val_indices = train_test_split(images, masks, \
                                                                                    masks_orig, good_training_images, \
                                                                                    range(len(good_training_images)), \
                                                                                     stratify=coverage_category, \
                                                                                     test_size=0.2)

        ##########################################################################################
        ##### Create all generators #############################################################
        ########################################################################################

        generator_dict = {'horizontal_flip': True}

        image_datagen = ImageDataGenerator(**generator_dict)
        mask_datagen = ImageDataGenerator(**generator_dict)

        image_generator = image_datagen.flow(
                train_images,
                batch_size=batch_size,
                seed=1)

        mask_generator = mask_datagen.flow(
                train_masks,
                batch_size=batch_size,
                seed=1)

        combined_generator = zip(image_generator, mask_generator)

        ### Load the current model ###
        model = unet_model(filter_scaling=filter_scaling, depth=depth, batch_norm_momentum=batch_norm_momentum, optimizer=optimizer)

        ### Name the model ###
        model_file = "./unet_models/unet_optimal_parameters_m1_to_1_optimizer_" + optimizer + "_fs_" + str(filter_scaling) + \
                     "_d_" + str(depth) + "_bs_" + str(batch_size) + "_bn_" + str(batch_norm_momentum) + "_fold_" + str(fold)
        if os.path.exists(model_file):
            raise ValueError("Model file already exists")

        ### Set up model training parameters ###
        early_stopping = EarlyStopping(patience=12, verbose=1)
        model_checkpoint = ModelCheckpoint(model_file, save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6, verbose=1)
        learning_rec= LearningRateHistory()

        ### Fit the model ###
        history = model.fit_generator(combined_generator, train_images.shape[0] // batch_size, \
                                      epochs=200, verbose=1, validation_data=(val_images, val_masks),
                                      callbacks=[early_stopping, model_checkpoint, reduce_lr, learning_rec])

        ### Load back in the best model ###
        model_to_load = model_file
        model.load_weights(model_to_load)

        ################################################################################
        ### Print out the history to file #############################################
        ##############################################################################
        df = pd.DataFrame()
        df['epoch'] = history.epoch
        df['loss'] = history.history['loss']
        df['acc'] = history.history['acc']
        df['val_loss'] = history.history['val_loss']
        df['val_acc'] = history.history['val_acc']
        df['lr'] = learning_rec.lr
        df.to_csv(model_file+'_history.csv',index=False)

        final_epoch = history.epoch[-1]

        ###############################################################################
        ### Extract best parameters ##################################################
        #############################################################################
        highest_val_acc = history.history['val_acc'][np.argmax(history.history['val_acc'])]
        lowest_val_loss = history.history['val_loss'][np.argmin(history.history['val_loss'])]

        ###############################################################################
        ### Do prediction without augmentation #######################################
        #############################################################################

        preds = model.predict(val_images)

        ### Use this when working with zero padded images ###
        preds_orig = np.array([smallenate_zero_pad(x) for x in preds.squeeze()])

        # Running on 10 images. Increase for better accuracy.
        thresholds = np.linspace(0, 1.0, 50)
        iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        mAPs = []
        for threshold in thresholds:
            APs = []
            for i in range(len(preds)):
                image = val_images[i]
                ### Need this if/else if you eliminated samples with sum()==0
                if image.sum()==0:
                    pred = np.zeros((101,101))
                    mask = val_orig_masks[i].squeeze().astype(np.int8)

                    AP = calc_avg_precision(pred, mask, threshold, iou_thresholds=iou_thresholds)
                else:
                    pred = preds_orig[i]
                    mask = val_orig_masks[i].squeeze().astype(np.int8)

                    AP = calc_avg_precision(pred, mask, threshold, iou_thresholds=iou_thresholds)

                APs.append(AP)

            mAPs.append(np.mean(APs))

        threshold_max = thresholds[np.argmax(mAPs)]
        mAP_max = mAPs[np.argmax(mAPs)]

        ###############################################################################
        ### Do prediction with augmentation ##########################################
        #############################################################################

        tta = TTA_ModelWrapper(model)
        preds = tta.predict(val_images)

        ### Use this when working with zero padded images ###
        preds_orig = np.array([smallenate_zero_pad(x) for x in preds.squeeze()])

        # Running on 10 images. Increase for better accuracy.
        thresholds = np.linspace(0, 1.0, 50)
        iou_thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        mAPs = []
        for threshold in thresholds:
            APs = []
            for i in range(len(preds)):
                image = val_images[i]
                ### Need this if/else if you eliminated samples with sum()==0
                if image.sum()==0:
                    pred = np.zeros((101,101))
                    mask = val_orig_masks[i].squeeze().astype(np.int8)

                    AP = calc_avg_precision(pred, mask, threshold, iou_thresholds=iou_thresholds)
                else:
                    pred = preds_orig[i]
                    mask = val_orig_masks[i].squeeze().astype(np.int8)

                    AP = calc_avg_precision(pred, mask, threshold, iou_thresholds=iou_thresholds)

                APs.append(AP)

            mAPs.append(np.mean(APs))

        threshold_max_augmentation = thresholds[np.argmax(mAPs)]
        mAP_max_augmentation = mAPs[np.argmax(mAPs)]

        #################################################################################################
        ##### Write the parameters out to file #########################################################
        ###############################################################################################

        output_vector = [filter_scaling,depth,batch_size,fold,batch_norm_momentum,optimizer,final_epoch,highest_val_acc,lowest_val_loss] + \
                        [threshold_max,mAP_max,threshold_max_augmentation,mAP_max_augmentation]
        output_string = ','.join(list(np.array(output_vector).astype('str')))

        with(open(record_file,'a')) as f:
            f.write('\n')
            f.write(output_string)






