# JCnet

This is a deep learning workflow for Progressive Multifocal Leukoencephalopathy lesion and brain segmentation from brain MRI using Python 3, Keras, and TensorFlow. The model generates segmentation masks of the brain parenchyma and lesions in PML patients. 
![TNS Logo](/assets/tns.jpg)

This work was developed by the Translation Neuroradiology Section (TNS) of the National Institute of Neurological Disorders and Stroke (NINDS), in collaboration with colleagues at the National Institute of Mental Health, Henry Jackson Foundation, and Radiology department at the National Institutes of Health. This software is distributed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

If this repository is helpful for your research, please cite the following articles:
*to be updated*

![JCnet](/assets/Figure-2.jpg)
The basic code structure was adopted from the following [source](https://www.nitrc.org/projects/flexconn/) (https://arxiv.org/abs/1803.09172) with several notable changes. We have created two tailored scripts for brain parechymal extraction and lesion segmentation training. We have also introduced several improvements in training/validation split which is now undertaken at the atlas level to remove patch sampling overlap effects, included support for Tensorboard logging, and generation of training/validation accuracy and loss graphs automatically at the end of model training. 
For the testing implementation, we fixed a previous bug with image padding, added a new 4D image padding function, and replaced the 'slice-by-slice' format previously used, with a new method to generate model predictions on unseen images using a moving 3D window, equal to the training 3D patch size, serially across the entire image volume to allow higher resolution images to fit into available GPU memory. In addition, the method now offers support for 3 different trainable network acrchitechures:
1. [3D Unet](https://arxiv.org/abs/1606.06650)
2. [Feature pyramid network-ResNet50 (with bottleneck ResNet modules)](https://arxiv.org/abs/1612.03144)
3. [Panoptic feature pyramid network-ResNet50 (with preactivated ResNet modules)](https://arxiv.org/abs/1901.02446)

## Prerequisites before running JCnet

### Pre-processing:
A few standard MRI preprocessing steps are necessary before training or testing a JCnet model:
1. Bias field correction - can use either [N4 bias correction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855/) or [MICO](https://www.sciencedirect.com/science/article/abs/pii/S0730725X14000927)
2. Skull-stripping - we recommend using the [MONSTR skull-stripping algorithm](https://pubmed.ncbi.nlm.nih.gov/27864083/) in PML cases, which is publicly available and can be found [here](https://www.nitrc.org/projects/monstr)
3. Co-registration of different MRI channels or contrasts (i.e. T1-weighted, fluid-attenuated inversion recovery, T2-weighted, and proton density images)

### Hardware Requirements:
1. Operating System: Linux
2. CPU Number/Speed: we recommend using a processor with at least 8 cores, 2GHz speed, and multithreading capability
3. RAM: 64+GB recommended (depending on the size of the training dataset and maximum number of training patches per subject)
4. GPU: recommend a dedicated graphics card with at least 8GB of VRAM (ex. NVIDIA RTX 2080 Ti, Titan X, or v100 models). If our current pre-trained models do not fit into GPU memory during testing, we recommend downscaling the network parameters (batch size, base filters, or patch size in this order). These models can be provided upon request.

### Software Requirements:
1. Python v3.6
2. Keras v2.2.4
3. Tensorflow GPU version v1.13+ (TF v2 is not currently supported)
4. Several open source python packages, please see 

### Training call examples

> hi there this is JCnet