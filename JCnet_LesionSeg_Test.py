from __future__ import print_function, division

import argparse
import multiprocessing
import os
# import shutil
import time
import sys
# import h5py
import statsmodels.api as sm
from scipy.signal import argrelextrema
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.models import load_model
from scipy import ndimage
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from keras.backend import tensorflow_backend as K
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
path = os.path.dirname(sys.argv[0])
path=os.path.abspath(path)
path=os.path.join(path,'CNNUtils')
sys.path.append(path)
from CNNUtils.DenseNet import layer2D, layer3D


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['OMP_NUM_THREADS']= '12'


def dice_coeff(y_true, y_pred):
    smooth = 0.001
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
    if contrast.lower() not in ['t1','t1c','t2','pd','fl','flc']:
        print("Contrast must be either T1,T1C,T2,PD,FL, or FLC. You entered %s. Returning original volume." % contrast)
        return vol
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

    # norm_vol = vol
    if contrast.lower() == "t1" or contrast.lower() == "t1c":
        print("Double checking peaks with a GMM.")
        gmm = GaussianMixture(n_components=3, covariance_type='spherical', tol=0.001,
                              reg_covar=1e-06, max_iter=100, n_init=2, init_params='kmeans',
                              weights_init=(0.33, 0.33, 0.34),
                              means_init=np.reshape((0.2 * q, 0.5 * q, 0.95 * q), (3, 1)), precisions_init=None,
                              warm_start=False,
                              verbose=0, verbose_interval=1)
        gmm.fit(temp.reshape(-1, 1))
        m=gmm.means_[2]
        peak = peaks[-1]
        if m/peak < 0.75 or m/peak > 1.25:
            print("WARNING: WM peak could be incorrect (%.4f vs %.4f). Please check." %(m,peak))
            peak = m
        print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol/peak
        # norm_vol[norm_vol > 1.25] = 1.25
        # norm_vol = norm_vol/1.25
    elif contrast.lower() in ['t2', 'pd', 'fl', 'flc']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol / peak
        # norm_vol[norm_vol > 3.5] = 3.5
        # norm_vol = norm_vol / 3.5
    else:
        print("Contrast must be either T1,T1C,T2,PD,FL, or FLC. You entered %s. Returning 1." % contrast)
    return vol/peak

def apply_model_3d(in_vol, psize, model):
    psize=np.asarray(psize,dtype=int)
    dsize = (np.array(psize)) / 2
    dsize=np.asarray(dsize,dtype=int)
    dim=in_vol.shape
    out_vol = np.zeros((dim[0],dim[1],dim[2]),dtype=np.float32)

    batch_dim = (1, psize[0], psize[1], psize[2], in_vol.shape[3])
    batch = np.zeros(batch_dim, dtype=np.float32)

    for z in tqdm(range(dsize[2], in_vol.shape[2] - dsize[2] - 1, dsize[2]*2 )):
        for y in range(dsize[1], in_vol.shape[1] - dsize[1] - 1, dsize[1]*2):
            for x in range(dsize[0], in_vol.shape[0] - dsize[0] - 1, dsize[0]*2):
                batch[0, :, :, :, :] = in_vol[x - dsize[0]:x + dsize[0], y-dsize[1] : y+dsize[1], z - dsize[2]:z + dsize[2], :]
                # print(batch.shape)
                # if np.ndarray.sum(batch) > 0:
                pred = model.predict_on_batch(batch)
                if type(pred) is list:  # To capture FPN, where output is multiple levels
                    pred1 = pred[0]
                else:
                    pred1 = pred
                out_vol[x - dsize[0]:x + dsize[0], y-dsize[1] : y+dsize[1], z - dsize[2]:z + dsize[2]] = pred1[0, :, :, :, 0]
    return out_vol


def apply_model_2d(in_vol, model):
    out_vol = np.zeros(in_vol.shape[:-1], dtype=np.float32)
    batch = np.zeros((1, in_vol.shape[0], in_vol.shape[1], in_vol.shape[3]), dtype=np.float32)

    for k in tqdm(range(in_vol.shape[2])):
        batch[0, :, :, :] = in_vol[:, :, k, :]
        # if np.ndarray.sum(invol) > 0: # For CT images, this does not hold. Will fix later.
        pred = model.predict_on_batch(batch)
        if type(pred) is list:  # To capture FPN, where output is multiple levels
            pred1 = pred[0]
        else:
            pred1 = pred
        out_vol[:, :, k] = pred1[0, :, :, 0]

    return out_vol

def pad_image(vol, padsize):
    dim = vol.shape
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim + 2 * padsize
    temp = np.zeros(dim2, dtype=np.float32)
    temp[padsize:dim[0] + padsize, padsize:dim[1] + padsize, padsize:dim[2] + padsize] = vol
    return temp


def split_filename(input_path):
    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)

    base_arr = basename.split('.')
    ext = ''
    if len(base_arr) > 1:
        ext = base_arr[-1]
        if ext == 'gz':
            ext = '.'.join(base_arr[-2:])
        ext = '.' + ext
        basename = basename[:-len(ext)]
    return dirname, basename, ext


def pad_image_4d(vol, padsize):
    dim = vol.shape
    numimgs = dim[-1]
    padsize = np.asarray(padsize, dtype=int)
    dim2 = dim[:3] + 2 * padsize
    dim2 = np.ceil(np.asarray(dim2, dtype=float) / 16) * 16  # For unet and vnet, image sizes must be multiple of 16
    dim2 = np.insert(np.asarray(dim2, dtype=int), 3, dim[-1])
    temp = np.zeros(dim2, dtype=np.float32)
    for j in range(numimgs):
        temp[padsize:dim[0] + padsize, padsize:dim[1] + padsize, padsize:dim[2] + padsize,j] = vol[: , : , : , j]
    return temp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JCnet Lesion Segmentation Testing')
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--models', type=str, nargs='+', required=True, help='Learnt models (.h5) files')
    required.add_argument('--images', type=str, nargs='+', required=True,
                        help='Images to find lesions. The order must be same as the order of atlases. '
                             'The atlas image order can be found from the trained models. Images should be in NIFTI '
                             '(.nii or .nii.gz). MR images are normalized to the white matter '
                             'peak based on the modalities provided. Images must be in the same orientation as the '
                             'atlases.')
    required.add_argument('--modalities', required=True, nargs='+',
                        help='A space separated string of input image modalities. The order must be same as the order '
                             'used during training. The order can be found from the trained model name also. '
                             'Accepted modalities are T1/T1C/T2/PD/FL/FLC. T1C and FLC corresponds to postcontrast T1 and FL.')
    required.add_argument('--psize', required=True, nargs='+', type=int,
                        help='Patch size, same one used for training. Can be seen in trained model names.')
    required.add_argument('--outdir', required=True,
                        help='Output directory where the resultant membership is written')
    optional = parser.add_argument_group('Optional arguments')
    optional.add_argument('--gpu', help='Specific GPU id to use. Default is 0.')
    optional.add_argument('--threshold', type=float, default=0.5,
                        help='Membership segmentation threshold (between 0 and 1). A reasonable number is 0.33. '
                             'Default is 0.5.')
    optional.add_argument('--withskull', action='store_true', default=False,
                          help='To check if the images are skullstripped or not. Default is False, i.e. skullstripped. '
                               'It is also an alternate way to stop re-scaling the default normalization based on contrast. '
                               'If the images have skull, then they must be pre-normalized because the default histogram '
                               'based normalization in --modalities option does not work. In that case, use --withskull '
                               'option. If this is chosen, then the images are not normalized again.')
    results = parser.parse_args()
    
    #numcpu = multiprocessing.cpu_count()
    #numcpu = int(numcpu / 2)
    for i in range(len(results.psize)):
        results.psize[i] = (results.psize[i]//2)*2
    
    if results.gpu is not None:
        print('Using gpu id %s' % results.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(results.gpu)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    elif results.gpu == 'none':
        os.environ['MKL_NUM_THREADS'] = '6'
        os.environ['GOTO_NUM_THREADS'] = '6'
        os.environ['OMP_NUM_THREADS'] = '6'
        os.eviron['openmp'] = 'True'
        with tf.Session(config=tf.ConfigProto(
                intra_op_parallelism_threads=6)) as sess:
            K.set_session(sess)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    else:
        print('Using gpu id 0')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


    for i in range(len(results.psize)):  # Patch size must be even
        results.psize[i] = (results.psize[i]//2)*2

    outdir = os.path.abspath(os.path.expanduser(results.outdir))
    
    results.models = [os.path.abspath(os.path.expanduser(model)) for model in results.models]
    print("%d models found at" % (len(results.models)))
    for model in results.models:
        print(model)
    
    #results.images = [os.path.abspath(os.path.expanduser(image)) for image in results.images]
    results.images = [os.path.expanduser(image) for image in results.images]

    _, base, _ = split_filename(results.images[0])
    membership_outname = os.path.join(outdir, base + "_CNNLesionMembership.nii.gz")
    mask_outname = os.path.join(outdir, base + "_CNNLesionMask.nii.gz")
    # results=['/data/allouzioa/MSv4/Batch1/FL_DS/MID001_t01_20130522_FL_DS.nii.gz', '/data/allouzioa/MSv4/Batch1/T1_DS/MID001_t01_20130522_T1_DS.nii.gz'];
    obj = nib.load(results.images[0])
    origdim = np.asarray(obj.shape,dtype=int)
    #paddeddim = (origdim//16+1)*16  # For unet and vnet, image sizes must be multiple of 16
    vol = np.zeros(obj.shape + (len(results.images),), dtype=np.float32)
    #vol = np.zeros(obj.shape + (len(results.images),), dtype=np.float32)
    outvol = np.zeros(obj.shape + (len(results.models),), dtype=np.float32)
    
    for i, img in enumerate(results.images):
        temp = nib.load(img).get_data().astype(np.float32)
        if results.withskull == False:
            contrast = results.modalities[i]
        else:
            contrast = 'unknown'

        if "XC.nii" in img or "YC.nii" in img or "ZC.nii" in img:
            temp=temp # Dont change image if spectral coordinate since they are already normalized
        else:
            temp = normalize_image(temp,contrast)
        vol[0:origdim[0], 0:origdim[1], 0:origdim[2], i] = temp

    psize = np.asarray(results.psize, dtype=int)
    padsize = np.max(np.asarray((psize//2), dtype=int))  # For unet and vnet, image sizes must be multiple of 16
    vol = pad_image_4d(vol, padsize)

    print('Original image size = %d x %d x %d ' % (origdim[0],origdim[1], origdim[2]))
    print("Padded matrix size = %d x %d x %d x %d" % vol.shape)

    dict = {"tf": tf,
            "dice_coeff": dice_coeff,
            "dice_coeff_loss": dice_coeff_loss,
            "focal_loss": focal_loss,
            "focal_loss_fixed": focal_loss_fixed,
            "layer2D": layer2D,
            'layer3D': layer3D,
            }
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}
    for i, model_file in enumerate(results.models):
        start = time.time()
        model = load_model(model_file, custom_objects=dict)

        try:
            #orient_code = os.path.basename(model_file).split('_')[4].replace('Orient', '')
            for x in os.path.basename(model_file).split('_'):
                if 'Orient' in x or 'orient' in x:
                    orient_code = x[6:9]
                    break
        except:
            orient_code = (0,1,2) # If orient_code is not found from the name, assume 012
            print("Image orientation code (e.g. 012 or 210) not found from model name. Assuming 012 (axial RAI).")


        transpose_code = tuple([int(dim) for dim in orient_code])
        reverse_codes = {(1, 2, 0): (2, 0, 1), (2, 0, 1): (1, 2, 0)}

        if transpose_code == (0, 1, 2):
            volorient = vol
        else:
            volorient = np.zeros_like(np.transpose(vol, transpose_code + (3, )))
            for j in range(len(results.images)):
                volorient[:, :, :, j] = np.transpose(vol[:, :, :, j], transpose_code)
        
        if len(results.psize) == 2:
            mem = apply_model_2d(volorient, model)
        else:
            mem = apply_model_3d(volorient, results.psize, model)

        if transpose_code != (0, 1, 2):
            mem = np.transpose(mem, reverse_codes[transpose_code])
        
        outvol[:, :, :, i] = mem[padsize:(origdim[0]+padsize), padsize:(origdim[1]+padsize), padsize:(origdim[2]+padsize)]
        elapsed = time.time() - start
        
        print("Time taken for %d%s atlas= %.2f seconds" % (i + 1, suffix.get(i+1, 'th'), elapsed))

    outvol = np.average(outvol, axis=3)

    # If the max membership is > 50, assume the loss was mse/mae, and then scale by 100. Simple way to
    # exclude the loss function in the testing script.
    print(np.max(np.ndarray.flatten(outvol)))
    if np.max(np.max(np.ndarray.flatten(outvol)))>=50:
        outvol = outvol / 100.0

    
    # save the whole membership
    print("Writing " + membership_outname)
    nib.Nifti1Image(outvol, obj.affine, obj.header).to_filename(membership_outname)
    
    print("Using threshold %.4f" % results.threshold)
    seg = np.zeros_like(outvol)
    seg[outvol >= results.threshold] = 1
    label = np.asarray(seg, dtype=np.uint8)
    '''
    se = ndimage.morphology.generate_binary_structure(3, 1)
    label, ncomp = ndimage.label(seg, structure=se)
    unique, counts = np.unique(label, return_counts=True)
    print("Using a 9 voxel volume threshold.")
    for j, unq in enumerate(unique):
        if counts[j] < 9:
            label[label == unq] = 0
    label[label > 0] = 1
    label=np.asarray(label,dtype=np.uint8)
    '''
    obj.header['bitpix']=8
    obj.header['datatype'] = 2
    print("Writing " + mask_outname)
    nib.Nifti1Image(label, obj.affine, obj.header).to_filename(mask_outname)
