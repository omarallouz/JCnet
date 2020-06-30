from __future__ import print_function, division

import tensorflow as tf # don't import tf within a def, it will cause model saving error when trained with multiple gpu
from keras.engine import Input, Model
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate,
                          Lambda, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D)

from keras.layers.merge import add
import keras.backend as K
K.set_image_data_format('channels_last')

def Unet2D(inputs, numfilter):

    conv1d = Conv2D(numfilter, (3, 3), activation='relu', padding='same', strides=(1, 1))(inputs)
    conv1d = Conv2D(numfilter, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv1d)

    pool1d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1d)

    conv2d = Conv2D(numfilter*2, (3, 3), activation='relu', padding='same', strides=(1, 1))(pool1d)
    conv2d = Conv2D(numfilter*2, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv2d)

    pool2d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2d)

    conv3d = Conv2D(numfilter*4, (3, 3), activation='relu', padding='same', strides=(1, 1))(pool2d)
    conv3d = Conv2D(numfilter*4, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv3d)

    pool3d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3d)

    conv4d = Conv2D(numfilter*8, (3, 3), activation='relu', padding='same', strides=(1, 1))(pool3d)
    conv4d = Conv2D(numfilter*8, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv4d)

    pool4d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4d)

    conv5d = Conv2D(numfilter*16, (3, 3), activation='relu', padding='same', strides=(1, 1))(pool4d)
    conv5d = Conv2D(numfilter*16, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv5d)

    up4u = UpSampling2D(size=(2, 2))(conv5d)

    conv4u = concatenate([conv4d, up4u], axis=-1)
    conv4u = Conv2D(numfilter*8, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv4u)
    conv4u = Conv2D(numfilter*8, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv4u)

    up3u = UpSampling2D(size=(2, 2))(conv4u)

    conv3u = concatenate([up3u, conv3d], axis=-1)
    conv3u = Conv2D(numfilter*4, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv3u)
    conv3u = Conv2D(numfilter*4, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv3u)

    up2u = UpSampling2D(size=(2, 2))(conv3u)

    conv2u = concatenate([up2u, conv2d], axis=-1)
    conv2u = Conv2D(numfilter*2, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv2u)
    conv2u = Conv2D(numfilter*2, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv2u)

    up1u = UpSampling2D(size=(2, 2))(conv2u)

    conv1u = concatenate([up1u, conv1d], axis=-1)
    conv1u = Conv2D(numfilter, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv1u)
    final = Conv2D(numfilter, (3, 3), activation='relu', padding='same', strides=(1, 1))(conv1u)
    return  final



def Unet3D(inputs, numfilter):

    conv1d = Conv3D(numfilter, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(inputs)
    conv1d = Conv3D(numfilter, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv1d)

    pool1d = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv1d)

    conv2d = Conv3D(numfilter*2, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(pool1d)
    conv2d = Conv3D(numfilter*2, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv2d)

    pool2d = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv2d)

    conv3d = Conv3D(numfilter*4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(pool2d)
    conv3d = Conv3D(numfilter*4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv3d)

    pool3d = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv3d)

    conv4d = Conv3D(numfilter*8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(pool3d)
    conv4d = Conv3D(numfilter*8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv4d)

    pool4d = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv4d)

    conv5d = Conv3D(numfilter*16, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(pool4d)
    conv5d = Conv3D(numfilter*16, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv5d)

    up4u = UpSampling3D(size=(2, 2, 2))(conv5d)

    conv4u = concatenate([conv4d, up4u], axis=-1)
    conv4u = Conv3D(numfilter*8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv4u)
    conv4u = Conv3D(numfilter*8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv4u)

    up3u = UpSampling3D(size=(2, 2, 2))(conv4u)

    conv3u = concatenate([up3u, conv3d], axis=-1)
    conv3u = Conv3D(numfilter*4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv3u)
    conv3u = Conv3D(numfilter*4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv3u)

    up2u = UpSampling3D(size=(2, 2, 2))(conv3u)

    conv2u = concatenate([up2u, conv2d], axis=-1)
    conv2u = Conv3D(numfilter*2, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv2u)
    conv2u = Conv3D(numfilter*2, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv2u)

    up1u = UpSampling3D(size=(2, 2, 2))(conv2u)

    conv1u = concatenate([up1u, conv1d], axis=-1)
    conv1u = Conv3D(numfilter, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv1u)
    final = Conv3D(numfilter, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv1u)
    return  final