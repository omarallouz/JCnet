# JCnet

This is a deep learning workflow for Progressive Multifocal Leukoencephalopathy lesion and brain segmentation from brain MRI using Python 3, Keras, and TensorFlow. The model generates segmentation masks of the brain parenchyma and lesions in PML patients. 
![TNS Logo](/assets/tns.jpg)
This work was developed by the Translation Neuroradiology Section (TNS), in collaboration with colleagues at the National Institute of Mental Health, Henry Jackson Foundation, and Radiology department at the National Institutes of Health. This software is distributed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

If this repository is helpful for your research, please cite the following article: *to be updated*

![JCnet](/assets/Figure-2.jpg)
The basic code structure was adopted from the following [source](https://www.nitrc.org/projects/flexconn/) (https://arxiv.org/abs/1803.09172) with several notable changes. We have created two tailored scripts for brain parechymal extraction and lesion segmentation training. We have also introduced improvements in training/validation split which is now undertaken at the atlas level to remove patch sampling overlap effects, included support for Tensorboard logging, and generation of training/validation accuracy and loss graphs automatically at the end of training. For the testing implementation, we fixed a previous bug with image padding, added a new 4D image padding function, and instead of the slice-by-slice format previously used, we now generate model predictions on unseen images using a moving 3D window, equal to the training 3D patch size, serially across the entire image volume to allow higher resolution images to fit into available GPU memory. In addition, the method now offers support for 3 different trainable network acrchitechures:
1. [3D Unet](https://arxiv.org/abs/1606.06650)
2. [Feature pyramid network-ResNet50 (with bottleneck ResNet modules)](https://arxiv.org/abs/1612.03144)
3. [Panoptic feature pyramid network-ResNet50 (with preactivated ResNet modules)](https://arxiv.org/abs/1901.02446)

<font size="+5">**Prerequisites before running JCnet**</font>

