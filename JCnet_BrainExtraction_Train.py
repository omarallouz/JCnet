from __future__ import print_function, division

import argparse
import os
import random
import time
import statsmodels.api as sm
import nibabel as nib
import numpy as np
import tensorflow as tf  # don't import tf within a def, it will cause model saving error when trained with multiple gpu
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import backend,losses
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.engine import Input, Model
from keras.layers.advanced_activations import PReLU, ReLU
from keras.layers import (AveragePooling2D, AveragePooling3D, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, concatenate,
                          Lambda, Conv2DTranspose, Conv3DTranspose, UpSampling3D, UpSampling2D, BatchNormalization, Activation, Add)
from keras.optimizers import Adam
from keras.layers.merge import add
from keras.regularizers import l2
from scipy import ndimage
from keras.models import load_model
import sys
from skimage.transform import resize
import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.signal import argrelextrema
path = os.path.dirname(sys.argv[0])
path=os.path.abspath(path)
path=os.path.join(path,'CNNUtils')
sys.path.append(path)
from CNNUtils.Inception import Inception2d, Inception3d
from CNNUtils.Unet import Unet2D,Unet3D
from CNNUtils.Vnet import Vnet2D,Vnet3D
from CNNUtils.FPN_preactivated import FPN2D, FPN3D
from CNNUtils.DenseNet import DenseNet3D,DenseNet2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

backend.set_floatx('float32')
backend.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ModelMGPU(Model):

    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        """
        Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        """
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def dice_coeff(y_true, y_pred):
    smooth = 0.0001
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coeff_loss(y_true, y_pred):
    return 1-dice_coeff(y_true, y_pred)

def focal_loss(gamma):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-6
        alpha = 0.5
        y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0),axis=-1)
    return focal_loss_fixed

def focal_loss_fixed(y_true, y_pred):
    gamma = 2
    eps = 1e-6
    alpha = 0.5
    y_pred=K.clip(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0),axis=-1)

def pad_image(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim + 2 * padsize
    temp = np.zeros(dim2, dtype=np.float32)
    temp[padsize:dim[0] + padsize, padsize:dim[1] + padsize, padsize:dim[2] + padsize] = vol
    return temp

def normalize_image(vol, contrast):
    # All MR images must be non-negative. Sometimes cubic interpolation may introduce negative numbers.
    # This will also affect if the image is CT, which not considered here. Non-negativity is required
    # while getting patches, where nonzero voxels are considered to collect patches.

    if contrast.lower() not in ['t1','t1c','t2','pd','fl','flc']:
        print("Contrast must be either T1,T1C,T2,PD,FL, or FLC. You entered %s. Returning original volume." % contrast)
        return vol

    vol[vol<0] = 0
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 1.00
    print("%d peaks found." % (len(peaks)))


    if contrast.lower() == "t1" or contrast.lower() == "t1c":
        print("Double checking peaks with a GMM.")
        gmm = GaussianMixture(n_components=3, covariance_type='spherical', tol=0.001,
                reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', precisions_init=None,
                weights_init=(0.33, 0.33, 0.34), means_init=np.reshape((0.2 * q, 0.5 * q, 0.95 * q), (3, 1)),
                warm_start=False, verbose=0, verbose_interval=1)
        gmm.fit(temp.reshape(-1, 1))
        m = gmm.means_[2]
        peak = peaks[-1]
        if m / peak < 0.75 or m / peak > 1.25:
            print("WARNING: WM peak could be incorrect (%.4f vs %.4f). Please check." % (m, peak))
            peaks = m
        peak = peaks[-1]
        print("Peak found at %.4f for %s" % (peak, contrast))
    elif contrast.lower() in ['t2', 'pd', 'fl', 'flc']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        print("Peak found at %.4f for %s" % (peak, contrast))
    else:
        print("Contrast must be either T1,T1C,T2,PD,FL, or FLC. You entered %s. Returning original volume." % contrast)
    return vol/peak

def resize_image(inarray,outarray,indx):
    dim = np.asarray(inarray.shape,dtype=int)

    if len(dim) == 5:
        d0=(dim[1],dim[2],dim[3])
        d0=np.asarray(d0, dtype=int)
        d0 = d0//2
        for c in range(0,dim[4]):
            outarray[indx, :, :, :, c] = resize(inarray[indx, :, :, :, c], output_shape=d0, order=0,
                                              mode='constant', anti_aliasing=True)
    else:
        d0 = (dim[1], dim[2])
        d0 = np.asarray(d0, dtype=int)
        d0 = d0 // 2
        for c in range(0,dim[3]):
            outarray[indx, :, :, c] = resize(inarray[indx, :, :, c], output_shape=d0, order=0,
                                              mode='constant', anti_aliasing=True)

def frac_eq_to(image, value=1):
    return (image == value).sum() / float(np.prod(image.shape))


def get_patches(vol4d, mask, opt, image_patches, mask_patches, count):
    patchsize = opt['patchsize']
    nummodal = len(opt['modalities'])
    maxpatch = opt['max_patches']
    patchsize = np.asarray(patchsize, dtype=int)
    dsize = np.floor(patchsize / 2).astype(dtype=int)
    mask = np.asarray(mask, dtype=np.float32)
    rng = random.SystemRandom()

    if opt['loss'] == 'mse' or opt['loss'] == 'mae':
        if len(patchsize) == 3:
            blurmask = ndimage.filters.gaussian_filter(mask, sigma=(1, 1, 1))
        else:
            blurmask = np.zeros(mask.shape, dtype=np.float32)
            for t in range(0, mask.shape[2]):
                if np.ndarray.sum(mask[:, :, t]) > 0:
                    blurmask[:, :, t] = ndimage.filters.gaussian_filter(mask[:, :, t], sigma=(1, 1))

        blurmask = np.ndarray.astype(blurmask, dtype=np.float32)
        blurmask[blurmask < 0.0001] = 0
        blurmask = blurmask * 100  # Just to have reasonable looking error values during training, no other reason.
    else:
        blurmask = mask

    indx = np.nonzero(mask)  # indx for positive patches
    indx = np.asarray(indx, dtype=int)

    num_patches = np.minimum(maxpatch, len(indx[0]))
    print('Number of patches used  = %d (out of %d, maximum %d)' % (num_patches, len(indx[0]), maxpatch))
    randindx = random.sample(range(0, len(indx[0])), num_patches)
    newindx = np.ndarray((3, num_patches))
    for i in range(0, num_patches):
        for j in range(0, 3):
            newindx[j, i] = indx[j, randindx[i]]
    newindx = np.asarray(newindx, dtype=int)

    # Add some negative samples as well
    # r = 1
    r = opt['oversample']  # negative patch oversampling ratio
    temp = copy.deepcopy(vol4d[:, :, :, 0])
    temp[temp > 0] = 1
    temp[temp <= 0] = 0
    temp = np.multiply(temp, 1 - mask)
    indx0 = np.nonzero(temp)
    indx0 = np.asarray(indx0, dtype=int)
    L = len(indx0[0])

    # Sample equal number of negative patches
    randindx0 = rng.sample(range(0, L), r * num_patches)
    newindx0 = np.ndarray((3, r * num_patches))
    for i in range(0, r * num_patches):
        for j in range(0, 3):
            newindx0[j, i] = indx0[j, randindx0[i]]
    newindx0 = np.asarray(newindx0, dtype=int)

    newindx = np.concatenate([newindx, newindx0], axis=1)

    num_patch_for_subj = newindx.shape[1]

    if len(patchsize) == 2:

        for i in range(0, (r + 1) * num_patches):
            idx1 = newindx[0, i]
            idx2 = newindx[1, i]
            idx3 = newindx[2, i]

            for m in range(0, nummodal):
                image_patches[count + i, :, :, m] = vol4d[idx1 - dsize[0]:idx1 + dsize[0],
                                                    idx2 - dsize[1]:idx2 + dsize[1], idx3, m]
            mask_patches[count + i, :, :, 0] = blurmask[idx1 - dsize[0]:idx1 + dsize[0],
                                               idx2 - dsize[1]:idx2 + dsize[1], idx3]
    else:

        for i in range(0, (r + 1) * num_patches):
            idx1 = newindx[0, i]
            idx2 = newindx[1, i]
            idx3 = newindx[2, i]

            for m in range(0, nummodal):
                image_patches[count + i, :, :, :, m] = vol4d[idx1 - dsize[0]:idx1 + dsize[0],
                                                       idx2 - dsize[1]:idx2 + dsize[1],
                                                       idx3 - dsize[2]:idx3 + dsize[2], m]
            mask_patches[count + i, :, :, :, 0] = blurmask[idx1 - dsize[0]:idx1 + dsize[0],
                                                  idx2 - dsize[1]:idx2 + dsize[1],
                                                  idx3 - dsize[2]:idx3 + dsize[2]]
    return num_patch_for_subj
    # return image_patches, mask_patches

def get_model_3d(kwargs):
    base_filters = kwargs['base_filters']
    gpus = kwargs['numgpu']
    loss = kwargs['loss']
    numchannel = int(len(kwargs['modalities']))
    inputs = Input((None, None, None, int(numchannel)))
    if kwargs['model'] == 'inception':
        conv1 = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(inputs)
        conv2 = Conv3D(base_filters * 8, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(conv1)
        inception1 = Inception3d(conv2, base_filters)
        inception2 = Inception3d(inception1, base_filters)
        inception3 = Inception3d(inception2, base_filters)
        convconcat1 = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(inception3)
        final = Conv3D(base_filters * 4, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(convconcat1)
    elif kwargs['model'] == 'unet':
        final = Unet3D(inputs, base_filters)
    elif kwargs['model'] == 'vnet':
        final = Vnet3D(inputs, base_filters)
    elif kwargs['model'] == 'fpn':
        reg = 0.0001
        f1, f2, f3, f4, _ = FPN3D(inputs, base_filters, reg)
    elif kwargs['model'] == 'densenet':
        final = DenseNet3D(inputs,base_filters)
    else:
        sys.exit('Model must be inception/unet/vnet/fpn.')

    if kwargs['model'] != 'fpn':
        if loss == 'bce'  or loss == 'dice' or loss == 'focal':
            final = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', strides=(1, 1, 1))(final)
        else:
            final = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(final)
        model = Model(inputs=inputs, outputs=final,name='some_unique_name')
    else:
        if loss == 'bce'  or loss == 'dice' or loss == 'focal':
            # Upsampling stages for F4
            # U1
            f4 = BatchNormalization(axis=-1)(f4)
            f4 = Activation('relu')(f4)
            f4 = UpSampling3D(size=(2, 2, 2), name='F4_U1')(f4)
            # U2
            f4 = Conv3D(base_filters*4, (3, 3, 3), padding='same', strides=(1, 1, 1), kernel_regularizer=l2(reg))(f4)
            f4 = BatchNormalization(axis=-1)(f4)
            f4 = Activation('relu')(f4)
            f4 = UpSampling3D(size=(2, 2, 2), name='F4_U2')(f4)
            # U3
            f4 = Conv3D(base_filters*4, (3, 3, 3), padding='same', strides=(1, 1, 1), kernel_regularizer=l2(reg))(f4)
            f4 = BatchNormalization(axis=-1)(f4)
            f4 = Activation('relu')(f4)
            f4 = UpSampling3D(size=(2, 2, 2), name='F4_U3')(f4)

            # Prepare
            f4 = Conv3D(base_filters*4, (3, 3, 3), padding='same', strides=(1, 1, 1), kernel_regularizer=l2(reg))(f4)
            f4 = BatchNormalization(axis=-1)(f4)
            f4 = Activation('relu')(f4)

            # Upsampling stages for F3
            # U1
            f3 = BatchNormalization(axis=-1)(f3)
            f3 = Activation('relu')(f3)
            f3 = UpSampling3D(size=(2, 2, 2), name='F3_U1')(f3)
            # U2
            f3 = Conv3D(base_filters*4, (3, 3, 3), padding='same', strides=(1, 1, 1), kernel_regularizer=l2(reg))(f3)
            f3 = BatchNormalization(axis=-1)(f3)
            f3 = Activation('relu')(f3)
            f3 = UpSampling3D(size=(2, 2, 2), name='F3_U2')(f3)
            # Prepare
            f3 = Conv3D(base_filters*4, (3, 3, 3), padding='same', strides=(1, 1, 1), kernel_regularizer=l2(reg))(f3)
            f3 = BatchNormalization(axis=-1)(f3)
            f3 = Activation('relu')(f3)

            # Upsampling stages for F2
            # U1
            f2 = BatchNormalization(axis=-1)(f2)
            f2 = Activation('relu')(f2)
            f2 = UpSampling3D(size=(2, 2, 2), name='F2_U1')(f2)
            # Prepare
            f2 = Conv3D(base_filters*4, (3, 3, 3), padding='same', strides=(1, 1, 1), kernel_regularizer=l2(reg))(f2)
            f2 = BatchNormalization(axis=-1)(f2)
            f2 = Activation('relu')(f2)

            # Prepare F1
            f1 = BatchNormalization(axis=-1)(f1)
            f1 = Activation('relu')(f1)

            f3 = Add()([f4, f3])
            f2 = Add()([f3, f2])
            f1 = Add()([f2, f1])

            f1 = Conv3D(base_filters*4, (3, 3, 3),  padding='same', strides=(1,1,1), kernel_regularizer=l2(reg))(f1)
            f1 = BatchNormalization(axis=-1)(f1)
            f1 = Activation('relu')(f1)
            final = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same', strides=(1, 1, 1), name='Level1')(f1)

        else:
            f1 = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(f1)
            f2 = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(f2)
            f3 = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(f3)
            f4 = Conv3D(1, (3, 3, 3), activation='relu', padding='same', strides=(1, 1, 1))(f4)

        model = Model(inputs=inputs, outputs=final,name='some_unique_name')
    #print(model.summary())
    return model

def check_nifti_filepath(directory, file_prefix):
    filepath = os.path.join(directory, file_prefix + '.nii.gz')
    filepath = filepath if os.path.exists(filepath) else os.path.join(directory, file_prefix + '.nii')
    if not os.path.exists(filepath):
        raise ValueError('File %s does not exists' % filepath)
    return filepath

def main(**kwargs):
    nummodal = int(len(kwargs['modalities']))
    padsize = np.max(np.array(kwargs['patchsize']) + 1) / 2

    subj_indx = range(kwargs['numatlas'])
    val_split = int(np.floor(kwargs['numatlas'] * 0.2))
    train_split = (kwargs['numatlas'] - val_split)
    # Randomly sample vaildation cases
    val_indx = random.sample(subj_indx, val_split)
    # Identify residual elements to be used for training
    train_indx = [x for x in subj_indx if x not in val_indx]

    kwargs['batchsize'] = kwargs['batchsize'] * len(kwargs['gpu_ids'])
    # Training set
    f = 0
    for i in train_indx:
        maskname = check_nifti_filepath(kwargs['atlasdir'], ('atlas%d' % (i + 1)) + '_' + 'bmask')
        f += min(kwargs['max_patches'], nib.load(maskname).get_data().sum())

    # Validation set
    v = 0
    for i in val_indx:
        maskname = check_nifti_filepath(kwargs['atlasdir'], ('atlas%d' % (i + 1)) + '_' + 'bmask')
        v += min(kwargs['max_patches'], nib.load(maskname).get_data().sum())

    print('Total number of Bmask patches = ' + str(int(f)))
    loss = kwargs['loss']
    r = opt['oversample'] # ratio between positive and negative patches
    if kwargs['model'] != 'fpn':
        print('Approximate memory required = %.1f GB' %(np.prod(kwargs['patchsize'])*(nummodal+1)*4.0*(r+1)*(f+v)/(1024**3)))
    else:
        x=np.prod(kwargs['patchsize'])*(nummodal+1)*4.0*(r+1)*(f+v)/(1024**3)
        if len(kwargs['patchsize'])==2:
            x = x + x/4.0 + x/16.0
        else:
            x = x + x / 8.0 + x / 64.0
        print('Approximate memory required = %.1f GB' % (x))


    time_id = time.strftime('%d-%m-%Y_%H-%M-%S')
    print('Unique ID is %s ' % time_id)
    con = '+'.join([str(mod).upper() for mod in kwargs['modalities']])
    psize = 'x'.join([str(side) for side in kwargs['patchsize']])
    outname = 'JCnet_BrainExtraction_Model_' + kwargs['model'].upper() + '_' + psize + '_Orient%d%d%d_' + con + '_' + time_id + '.h5'

    train_mask_patches = np.zeros((int((r + 1) * f),) + kwargs['patchsize'] + (1,), dtype=np.float32)
    train_image_patches = np.zeros((int((r + 1) * f),) + kwargs['patchsize'] + (nummodal,), dtype=np.float32)

    val_mask_patches = np.zeros((int((r + 1) * v),) + kwargs['patchsize'] + (1,), dtype=np.float32)
    val_image_patches = np.zeros((int((r + 1) * v),) + kwargs['patchsize'] + (nummodal,), dtype=np.float32)

    if kwargs['model'] == 'fpn':
        if len(kwargs['patchsize'])==2:
            dim = np.asarray(train_mask_patches.shape, dtype=int)
            dim2 = (dim[0], dim[1] / 2, dim[2] / 2, dim[3])
            dim2 = np.asarray(dim2, dtype=int)
            dim3 = (dim[0], dim[1] / 4, dim[2] / 4, dim[3])
            dim3 = np.asarray(dim3, dtype=int)
            dim4 = (dim[0], dim[1] / 8, dim[2] / 8, dim[3])
            dim4 = np.asarray(dim4, dtype=int)
            dim5 = (dim[0], dim[1] / 16, dim[2] / 16, dim[3])
            dim5 = np.asarray(dim5, dtype=int)
            mask_patches2 = np.zeros(dim2, dtype=np.float32)
            mask_patches3 = np.zeros(dim3, dtype=np.float32)
            mask_patches4 = np.zeros(dim4, dtype=np.float32)
            mask_patches5 = np.zeros(dim5, dtype=np.float32)
        else:
        # Predefine the big matrices here, so that the memory isn't allocated repeatedly for three orientations
            dim = np.asarray(train_mask_patches.shape, dtype=int)
            dim2 = (dim[0], dim[1] / 2, dim[2] / 2, dim[3] / 2, dim[4])
            dim2 = np.asarray(dim2, dtype=int)
            dim3 = (dim[0], dim[1] / 4, dim[2] / 4, dim[3] / 4, dim[4])
            dim3 = np.asarray(dim3, dtype=int)
            dim4 = (dim[0], dim[1] / 8, dim[2] / 8, dim[3] / 8, dim[4])
            dim4 = np.asarray(dim4, dtype=int)
            mask_patches2 = np.zeros(dim2, dtype=np.float32)
            mask_patches3 = np.zeros(dim3, dtype=np.float32)
            mask_patches4 = np.zeros(dim4, dtype=np.float32)

    if len(kwargs['patchsize']) == 2:
        opt2 = copy.deepcopy(kwargs)
        opt2['numgpu'] = 1
        model = get_model_2d(opt2)
    else:
        opt2 = copy.deepcopy(kwargs)
        opt2['numgpu'] = 1
        model = get_model_3d(opt2)

    if opt['initmodel'] != 'None' and os.path.exists(opt['initmodel']):
        dict = {"tf": tf,
                "dice_coeff": dice_coeff,
                "dice_coeff_loss": dice_coeff_loss,
                "focal_loss": focal_loss,
                "focal_loss_fixed": focal_loss_fixed,
                }
        try:
            oldmodel = load_model(opt['initmodel'], custom_objects=dict)
            model.set_weights(oldmodel.get_weights())
            print("Initialized from existing model %s" % (opt['initmodel']))
        except Exception as e:
            print('ERROR: Can not load from pre-trained model ' + opt['initmodel'])
            print(str(e))

    if kwargs['numgpu'] > 1:
        model = ModelMGPU(model,kwargs['numgpu'])


    codes = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    
    for orient in range(1 if kwargs['axial_only'] else 3):

        # Recompiling the model with appropriate loss and learning rate is needed at the beginning.
        # If this is done outside the orient loop, then the lr keeps decreasing, if the early_stopping
        # criteria is used during (0,1,2) orientation. Recompiling does not change model values though,
        # so during orient (1,2,0) and (2,0,1), the learnt parameters from (0,1,2) are used.
        #print('Total number of parameters = ' + str(model.count_params()))
        print("Total number of parameters = {:,}".format(model.count_params()))
        #if kwargs['model'] != 'fpn':
        if kwargs['numgpu'] == 1:
            if loss == 'bce':
                model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
            elif loss == 'mae':
                model.compile(optimizer=Adam(lr=0.0001), loss='mean_absolute_error')
            elif loss == 'mse':
                model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
            elif loss == 'focal':
                model.compile(optimizer=Adam(lr=1e-4), loss=[focal_loss(gamma=kwargs['gamma'])],
                              metrics=['accuracy'])
            else:
                model.compile(optimizer=Adam(lr=1e-4), loss=dice_coeff_loss, metrics=[dice_coeff])
        else:
            if loss == 'bce':
                model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
            elif loss == 'mae':
                model.compile(optimizer=Adam(lr=0.0001), loss='mean_absolute_error')
            elif loss == 'mse':
                model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
            elif loss == 'focal':
                model.compile(optimizer=Adam(lr=1e-4), loss=[focal_loss(gamma=kwargs['gamma'])],
                              metrics=['accuracy'])
            else:
                model.compile(optimizer=Adam(lr=1e-4), loss=dice_coeff_loss, metrics=[dice_coeff])

        transpose_code = codes[orient]
        orient_outname = os.path.join(kwargs['outdir'], outname % transpose_code)
        if kwargs['loss'] == 'bce' or kwargs['loss'] == 'focal':
            tempoutname = orient_outname.replace('.h5', '_epoch-{epoch:03d}_val_acc-{val_acc:.4f}.h5')
        elif kwargs['loss'] == 'dice':
            tempoutname = orient_outname.replace('.h5', '_epoch-{epoch:03d}_val_dice-{val_dice_coeff:.4f}.h5')
        else:
            tempoutname = orient_outname.replace('.h5', '_epoch-{epoch:03d}_val_loss-{val_loss:.4f}.h5')


        print('Model for orientation %s will be written at %s' % (str(transpose_code), orient_outname))

        train_patch_count = int(0)
        val_patch_count = int(0)
        for i in range(kwargs['numatlas']):
            #segpath = check_nifti_filepath(kwargs['atlasdir'], ('atlas%02d' % (i + 1)) + '_' + 'bmask')
            segpath = check_nifti_filepath(kwargs['atlasdir'], ('atlas%d' % (i + 1)) + '_' + 'bmask') # Original
            mask = np.transpose(pad_image(nib.load(segpath).get_data(), padsize),
                                axes=transpose_code).astype(np.float32)
            vol4d = np.zeros(mask.shape + (nummodal,), dtype=np.float32)
            
            for j in range(nummodal):
                filepath = check_nifti_filepath(kwargs['atlasdir'],
                                                ('atlas%d' % (i + 1)) + '_' + kwargs['modalities'][j].upper())
                #                                ('atlas%02d' % (i + 1)) + '_' + kwargs['modalities'][j].lower())
                print('Reading %s' % filepath)
                if kwargs['withskull'] == False:
                    contrast = kwargs['modalities'][j]
                else:
                    contrast = 'unknown'

                if kwargs['modalities'][j] == 'XC' or kwargs['modalities'][j] == 'YC' or kwargs['modalities'][
                    j] == 'ZC':
                    vol4d[:, :, :, j] = np.transpose(pad_image(nib.load(filepath).get_data()
                                                                            , padsize),
                                                     axes=transpose_code).astype(np.float32)
                else:
                    vol4d[:, :, :, j] = np.transpose(pad_image(normalize_image(nib.load(filepath).get_data(),
                                                                               contrast), padsize),
                                                     axes=transpose_code).astype(np.float32)

                #vol4d[:, :, :, j] = np.transpose(pad_image(normalize_image(nib.load(filepath).get_data(),
                #                    kwargs['modalities'][j]), padsize), axes=transpose_code).astype(np.float32)
            print('Atlas %d size = %s ' % (i+1 , str(vol4d.shape)))

            if i in train_indx:
                print('Atlas %d will be added to the training set' % (i + 1))
                num_patch_for_subj = get_patches(vol4d, mask, kwargs, train_image_patches, train_mask_patches, train_patch_count)
                print('Atlas %d : indices [%d,%d)' % (i + 1, train_patch_count, train_patch_count + num_patch_for_subj))
                train_patch_count += num_patch_for_subj
            elif i in val_indx:
                print('Atlas %d will be added to the validation set' % (i + 1))
                num_patch_for_subj = get_patches(vol4d, mask, kwargs, val_image_patches, val_mask_patches, val_patch_count)
                print('Atlas %d : indices [%d,%d)' % (i + 1, val_patch_count, val_patch_count + num_patch_for_subj))
                val_patch_count += num_patch_for_subj

            #num_patches = min(kwargs['max_patches'], mask.sum())
            #num_patches = int((r+1)*num_patches)
            print('-' * 100)

        print('Total number of patches collected = ' + str(train_patch_count + val_patch_count))
        print('Sizes of the training input matrices are ' + str(train_image_patches.shape) + ' and ' + str(train_mask_patches.shape))
        print('Sizes of the validation input matrices are ' + str(val_image_patches.shape) + ' and ' + str(val_mask_patches.shape))

        if kwargs['loss'] == 'bce' or kwargs['loss'] == 'focal':
            callbacks = [ModelCheckpoint(tempoutname, monitor='val_acc', verbose=1, save_best_only=True,
                                         period=kwargs['period'], mode='max')] if kwargs['period'] > 0 else None
            dlr = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=20,
                                    mode='max', verbose=1, cooldown=1, min_lr=1e-8)
            tensorboard = TensorBoard(log_dir=kwargs['outdir'] + '/logs')
            # earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=4,
                                      # verbose=1, mode='max')
        elif kwargs['loss'] == 'dice':
            callbacks = [ModelCheckpoint(tempoutname, monitor='val_dice_coeff', verbose=1, save_best_only=True,
                                         period=kwargs['period'], mode='max')] if kwargs['period'] > 0 else None
            dlr = ReduceLROnPlateau(monitor="val_dice_coeff", factor=0.5, patience=20,
                                    mode='max', verbose=1, cooldown=1, min_lr=1e-8)
            tensorboard = TensorBoard(log_dir=kwargs['outdir'] + '/logs')
            # earlystop = EarlyStopping(monitor='val_dice_coeff', min_delta=0.0001, patience=4,
                                      # verbose=1, mode='max')
        else:
            callbacks = [ModelCheckpoint(tempoutname, monitor='val_loss', verbose=1, save_best_only=True,
                                         period=kwargs['period'], mode='min')] if kwargs['period'] > 0 else None
            dlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20,
                                    mode='min', verbose=1, cooldown=1, min_lr=1e-8)
            tensorboard = TensorBoard(log_dir=kwargs['outdir'] + '/logs')
            # earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4,
                                      # verbose=1, mode='min')

        callbacks.append(dlr)
        callbacks.append(tensorboard)
        # callbacks.append(earlystop)

        """
        Keras crashes when using multi_gpu_model and n_sample is not a multiple of batch_size
        https://github.com/keras-team/keras/issues/11434     
        There are some weird errors like,
        ensorflow/stream_executor/cuda/cuda_dnn.cc:521] Check failed: cudnnSetTensorNdDescriptor(handle_.get(), 
        elem_type, nd, dims.data(), strides.data()) == CUDNN_STATUS_SUCCESS (3 vs. 0)batch_descriptor: {count: 0 
        feature_map_count: 24 spatial: 32 32 32  value_min: 0.000000 value_max: 0.000000 layout: BatchDepthYX}        
        """
        N = train_image_patches.shape[0]
        N = ((N//5)//kwargs['batchsize']) * kwargs['batchsize'] * 5
        N = np.asarray(N, dtype=int)
        
        M = val_image_patches.shape[0]
        M = ((M//5)//kwargs['batchsize']) * kwargs['batchsize'] * 5
        M = np.asarray(M, dtype=int)

        print('Using %d patches instead of total %d patches to train. ' % (N, train_image_patches.shape[0]))
        print('Using %d patches instead of total %d patches to validate. ' % (M, val_image_patches.shape[0]))
        print('Kera multi_gpu_model occasionally crashes if the number of samples is not a multiple of batchsize.')

        if len(kwargs['patchsize']) == 2:
            history = model.fit(image_patches[0:N,:,:,:], mask_patches[0:N,:,:,:], batch_size=kwargs['batchsize'],
                      epochs=kwargs['epoch'], verbose=1, validation_split=0.2, callbacks=callbacks, shuffle=True)
        else:
            history = model.fit(train_image_patches[0:N, :, :, :, :], train_mask_patches[0:N, :, :, :, :], batch_size=kwargs['batchsize'],
                      epochs=kwargs['epoch'], verbose=1, validation_data=(val_image_patches[0:M, :, :, :, :],val_mask_patches[0:M, :, :, :, :]), callbacks=callbacks, shuffle=True)

        print('Final model is written at ' + orient_outname)
        model.save(filepath=orient_outname)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(orient_outname.replace('.h5', '_accuracy.png'))
        plt.clf()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(orient_outname.replace('.h5', '_loss.png'))
        plt.clf()

        '''
        if kwargs['numgpu'] > 1:
            opt2 = copy.deepcopy(kwargs)
            opt2['numgpu'] = 1
            with tf.device("/cpu:0"):
                if len(kwargs['patchsize'])==2:
                    single_model = get_model_2d(opt2)
                else:
                    single_model = get_model_3d(opt2)


            single_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
            single_model.set_weights(model.get_weights())
            single_model.save(filepath=orient_outname)
            print(single_model.summary())
        else:
            model.save(filepath=orient_outname)
        #model.save(filepath=orient_outname)
        '''
        """
        # Beginning from keras 2.2.4, model.save writes the single gpu model by default, which does not work during prediction
        if len(kwargs['gpu_ids']) > 1:
            '''
            Training with multi-gpu model has a bug where the trained model can not be loaded to a single gpu for
            testing, giving some zero batch size errors, such as
                1) could not convert BatchDescriptor {count: 0 ...
                2) cudnn tensor descriptor: CUDNN_STATUS_BAD_PARAM
            Therefore the following fix works when models trained with multi-gpu are used for single gpu.
            It is obtained from the following link,
               https://stackoverflow.com/questions/47210811/can-not-save-model-using-model-save-following-multi-
               gpu-model-in-keras/48066771#48066771
            ** This is fixed in Keras 2.1.5, although it is better for backwards compatibility **
            ** Don't use default multi_gpu_model from Keras 2.1.5, sometimes the resulting model file can't be read. 
            Not sure why.**
            '''
            new_kwargs = copy.deepcopy(kwargs)
            new_kwargs['numgpu'] = 1
            with tf.device("/cpu:0"):
                single_model = get_model_2d(new_kwargs) if len(kwargs['patchsize']) == 2 \
                    else get_model_3d(new_kwargs)
            '''    
            if kwargs['loss'] == 'bce':
                single_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
            elif kwargs['loss'] == 'dice' :
                single_model.compile(optimizer=Adam(lr=0.0001), loss=dice_coeff_loss, metrics=[dice_coeff])
            elif loss == 'focal':
                single_model.compile(optimizer=Adam(lr=1e-4), loss=[focal_loss(gamma=kwargs['gamma'])], metrics=['accuracy'])
            else:
                single_model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error')
            '''
            single_model.set_weights(model.get_weights())
            single_model.save(filepath=orient_outname)
        else:
            model.save(filepath=orient_outname)
        """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JCnet Brain Extraction Training')
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--atlasdir', required=True,
                        help='Directory containing atlas images. Images should be in NIFTI (.nii or .nii.gz) and be '
                             'N4 bias corrected. Atlas images should be in the same orientation as the subject'
                             '(axial [RAI]). Atlases should be named atlas{NUMBER}_{MODALITY}.nii.gz. '
                             '{MODALITY} should match the strings entered in the "--modalities" argument. Example '
                             'atlas image names are atlas1_T1.nii.gz, atlas1_T2.nii.gz, atlas1_FL.nii.gz, atlas1_bmask.nii.gz, with '
                             'modalities as --modalities t1 t2 fl.')
    required.add_argument('--natlas', required=True, type=int,
                        help='Number of atlases to be used. The program will pick the first N atlases from the '
                             'atlas directory. The atlas directory must contain at least this many atlas sets.')
    required.add_argument('--psize', nargs='+', type=int, default=[64, 64, 64],
                        help='Patch size, e.g. 32 32 32 (3D) or 64 64 (2D). Patch sizes are separated by space. '
                             'Note that bigger patches (such as 128x128) are possible in 2D models while it is '
                             'computationally expensive to use more than 80x80x80 patches. Default is [64,64,64]. '
                             'For Unet/Vnet, patch sizes must be multiple of 16.')
    required.add_argument('--modalities', required=True, nargs='+',
                        help='A space separated string of input image modalities. This is used to determine the order '
                             'of images used in training. It also defines how many modalities will be used by training '
                             '(if the atlas directory contains more). Accepted modalities are T1/T1C/T2/PD/FL/FLC, in '
                             'addition to X-, Y-, and Z-axes normalized coordinate files. T1C '
                             'and FLC corresponds to postcontrast T1 and FL. This is also used to normalize images. '
                             'See --withskull in optional arguments if the images have skull, because normalization '
                             'based on contrast does not work when images have skull.')
    required.add_argument('--model', required=True, type=str, default='inception',
                        help='Training model type, options are Inception, Unet, Vnet, Densenet, and FPN. ')
    required.add_argument('--outdir', required=True, type=str,
                        help='Output directory where the trained models are written.')
    # Optional arguments
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--batchsize', type=int, default=32, required=False,
                        help='Mini-batch size to use per iteration. Usually 32-256 works well, '
                             '32 is used as default. ')
    optional.add_argument('--epoch', type=int, default=20, required=False,
                        help='Number of epochs to train. Usually 20-50 works well. Optional argumet, '
                             'if omitted, 20 is used as default. Too large epochs can incur in overfitting.')
    optional.add_argument('--save', type=int, default=1, required=False,
                        help='When training with large number of epochs, the interim models can be saved after every N '
                             'epochs. This option (e.g., --save 4) enables saving the model every 4th epoch.')
    optional.add_argument('--axialonly', action='store_true', default=False,
                        help='Only train a modal in the input (normally axial) orientation. This is common for CT '
                             'where images can be highly anisotropic. This also works for very thick-sliced MR images. '
                             'Without this option, the training images are reoriented in all 3 orientations and '
                             'individual trainings are done for each orientation separately.')
    optional.add_argument('--numgpu', type=int, default=1, required=False,
                        help='Number of GPUs to use for training. The program will use the first N visible GPUs. To '
                             'select specific gpus, use "--gpuids" ')
    optional.add_argument('--gpuids', type=int, nargs='+', required=False,
                        help='Specifc GPUs to use for training, separated by space. E.g., --gpuids 3 4 5 ')
    optional.add_argument('--basefilters', type=int, default=8,
                        help='Sets the base number of filters for the models. 16 is appropriate for 12GB GPUs, where '
                             '8 may be more appropriate for 4GB cards. This value scales all filter banks (increasing '
                             'by 2 for a change from 8->16). This '
                             'value is abitrary and should be scaled to fit the available GPU memory.')
    optional.add_argument('--maxpatches', type=int, default=10000, required=False,
                        help='Maximum number of patches to choose from each atlas. 100000 is the default. This '
                             'is appropriate for 2D patches. 10000 may be more appropriate for 3D patches. This '
                             'value is abitrary and should be scaled to fit the available memory.')
    optional.add_argument('--loss', type=str, default='bce',
                        help="Loss function to be used during training. Available options are mae (mean absolute error), "
                             "mse(mean squared error), Dice, focal, and bce (binary cross-entropy). If mse/mae are chosen, the "
                             "binary brain masks are first blurred by a Gaussian to compute a membership. If model is "
                             "FPN (feature pyramid networks), loss must be either Dice or BCE.")
    optional.add_argument('--initmodel', type=str, dest='INITMODEL', required=False,
                        help='Existing pre-trained model. If provided, the weights from the pre-trained model will be '
                             'used to initiate the training.')
    optional.add_argument('--oversample', type=int, dest='OVERSAMPLE', required=False, default=1,
                        help='Negative to positive patch ratio, i.e. oversampling negative patches. Default is 1, '
                             'indicating equal number of positive and negative patches are used in training. Use >1 '
                             'to have more negative patches to keep false positives down.')
    optional.add_argument('--withskull', action='store_true', default=False,
                          help='To check if the images are skullstripped or not. Default is False, i.e. skullstripped. '
                               'It is also an alternate way to stop re-scaling the default normalization based on contrast. '
                               'If the images have skull, then they must be pre-normalized because the default histogram '
                               'based normalization (given in --modalities) option does not work. In that case, use --withskull '
                               'option. If this is chosen, then the images are not normalized again.')
    results = parser.parse_args()


    if results.gpuids is not None:
        gpu_ids = results.gpuids
    else:
        gpu_ids = range(results.numgpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu_id) for gpu_id in gpu_ids])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    numgpu = len(gpu_ids)
    # results.batchsize = (results.batchsize // numgpu) * numgpu
    # Patch size must be even and multiple of 16 for unet/vnet
    for i in range(len(results.psize)):
        results.psize[i] = (results.psize[i]//2)*2


    if results.INITMODEL is None:
        results.INITMODEL = 'None'

    loss=['bce','mae', 'mse', 'dice', 'focal']
    if str(results.loss).lower() not in loss:
        sys.exit('Available loss options are MAE, MSE, Dice, Focal,  '
              'and BCE. You entered %s.' %(results.loss))

    model = ['inception', 'unet', 'vnet', 'fpn', 'densenet']
    if str(results.model).lower() not in model:
        sys.exit('Error: Available training model options are Inception, Unet, Densenet, and Vnet.  '
                 'You entered %s.' % (results.model))
    if results.model.lower() == 'unet' or results.model.lower() == 'vnet':
        if np.sum(np.mod(results.psize,16)) != 0:
            sys.exit('Error: Patch sizes must be multiple of 16 for Unet and Vnet')

    opt = {'numatlas': results.natlas,
           'outdir': os.path.abspath(os.path.expanduser(results.outdir)),
           'modalities': [item.upper() for item in results.modalities],
           'patchsize': tuple(results.psize),
           'atlasdir': os.path.abspath(os.path.expanduser(results.atlasdir)),
           'batchsize': results.batchsize,
           'epoch': results.epoch,
           'period': results.save,
           'model': str(results.model).lower(),
           'axial_only': results.axialonly,
           'withskull': results.withskull,
           'base_filters': results.basefilters,
           'max_patches': results.maxpatches,
           'gpu_ids': gpu_ids,
           'numgpu': numgpu,
           'loss': str(results.loss).lower(),
           'initmodel': results.INITMODEL,
           'oversample': results.OVERSAMPLE,
           }

    if opt['loss'].lower() == 'focal':
        opt['gamma'] = 2.  # Gamma parameter (>0) for focal loss, higher number penalizes ones more than zeros

    if not os.path.isdir(opt['outdir']):
        print('Output directory does not exist. I will create it.')
        os.makedirs(opt['outdir'])
    
    print('Atlas Directory     =', str(opt['atlasdir']))
    print('Number of atlases   =', str(opt['numatlas']))
    print('Training model      =', str(opt['model']).upper())
    print('Patch size          =', str(opt['patchsize']))
    print('Modalities          =', str(opt['modalities']))
    print('# Modalities        =', str(len(opt['modalities'])))
    print('Output directory    =', str(opt['outdir']))
    print('Batch size          =', str(opt['batchsize']))
    print('# Epochs            =', str(opt['epoch']))
    print('# Base Filters      =', str(opt['base_filters']))
    print('Max # Patches       =', str(opt['max_patches']))
    print('Do Multi-Orient     =', str(not opt['axial_only']))
    print('Is there skull?     =', str(opt['withskull']))
    print('Use Multiple GPUs   =', str(len(opt['gpu_ids']) > 1))
    print('GPU IDs             =', str(opt['gpu_ids']))
    print('Number of GPUs      =', str(opt['numgpu']))
    print('Training loss       =', str(opt['loss']).upper())
    print('Initial Model       =', str(opt['initmodel']))
    print('Oversampling ratio  =', str(opt['oversample']))
    if results.save > 0:
        print('Save every N epochs =', str(results.save))
    
    main(**opt)
