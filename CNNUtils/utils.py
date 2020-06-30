import tensorflow as tf
import numpy as np
import os
import sys
import statsmodels.api as sm
import keras.backend as K
from keras.utils import multi_gpu_model
from scipy.signal import argrelextrema
from keras.engine import Input, Model


class ModelMGPU(Model):

    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''
        Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
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



def Crop(vol,bg=0,padsize=0):
    dim=vol.shape
    cpparams=[]
    s = 0
    for k in range(0, dim[0]):
        if np.sum(vol[k, :, :]) == bg:
            s = s + 1
        else:
            break
    cpparams.append(np.maximum(s-padsize,0))
    s = dim[0]-1
    for k in range(dim[0]-1, -1, -1):
        if np.sum(vol[k, :, :]) == bg:
            s = s - 1
        else:
            break
    cpparams.append(np.minimum(s+padsize,dim[0]-1))
    s = 0
    for k in range(0, dim[1]):
        if np.sum(vol[:, k, :]) == bg:
            s = s + 1
        else:
            break
    cpparams.append(np.maximum(s - padsize, 0))
    s = dim[1]-1
    for k in range(dim[1]-1, -1, -1):
        if np.sum(vol[:, k, :]) == bg:
            s = s - 1
        else:
            break
    cpparams.append(np.minimum(s + padsize, dim[1]-1))
    s = 0
    for k in range(0, dim[2]):
        if np.sum(vol[:, :, k]) == bg:
            s = s + 1
        else:
            break
    cpparams.append(np.maximum(s - padsize, 0))
    s = dim[2]-1
    for k in range(dim[2]-1, -1, -1):
        if np.sum(vol[:, :, k]) == bg:
            s = s - 1
        else:
            break
    cpparams.append(np.minimum(s + padsize, dim[2]-1))
    vol2=vol[cpparams[0]:cpparams[1],cpparams[2]:cpparams[3],cpparams[4]:cpparams[5]]
    return vol2,cpparams,dim

def pad_image(vol,padsize=padsize, dim=3):
    '''
    :param vol: 3D volume
    :param padsize: a scalar int
    :param dim: either pad 3D equally with padsize, or only 2D slices, useful for large 2D patches, where padding all
    three dimensions may incur in large memory
    :return:
    '''

    if dim == 3:
        padsize=(padsize,padsize,padsize)
    else:
        padsize = (padsize, padsize, 0,)
    padsize=np.asarray(padsize,dtype=int)
    dim2=dim + 2*padsize
    temp=np.zeros(dim2,dtype=np.float32)
    temp[padsize[0]:dim[0]+padsize[0], padsize[1]:dim[1]+padsize[1], padsize[2]:dim[2]+padsize[2]]=vol
    return temp

