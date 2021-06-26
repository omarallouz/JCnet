# JCnet

This is a deep learning workflow for Progressive Multifocal Leukoencephalopathy (PML) lesion and brain segmentation from brain MRI using Python 3, Keras, and TensorFlow. The model generates segmentation masks of the brain parenchyma and lesions in PML patients. 

![TNS Logo](/assets/tns.jpg)

This work was conducted at the Neuroimmunology Clinic (NIC) and Translation Neuroradiology Section (TNS) of the National Institute of Neurological Disorders and Stroke (NINDS) at the National Institutes of Health. 
This work is supported by the NINDS Intramural research program and a National Multiple Sclerosis Society (NMSS) and American Brain Foundation (ABF) Clinician Scientist Development grant to O.A. (FAN-1807-32163). This work also utilized the computational resources of the [NIH HPC Biowulf cluster](http://hpc.nih.gov), and the software is distributed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).

If this repository is helpful for your research, please cite the following articles:

*Al-Louzi O, Roy S, Osuorah I, et al. Progressive multifocal leukoencephalopathy lesion and brain parenchymal segmentation from MRI using serial deep convolutional neural networks. Neuroimage Clin. 2020;28:102499. doi:10.1016/j.nicl.2020.102499*
[https://www.sciencedirect.com/science/article/pii/S2213158220303363](https://www.sciencedirect.com/science/article/pii/S2213158220303363)

![JCnet](/assets/Figure-2.jpg)

The basic code skeleton was adopted from the following [source](https://www.nitrc.org/projects/flexconn/) (https://arxiv.org/abs/1803.09172) with several notable changes. We have created two tailored scripts for PML brain parechymal extraction and lesion segmentation training. We have also introduced improvements in training/validation split which is now undertaken at the atlas level to remove patch overlap effects during sampling, included support for Tensorboard logging, and generation of training/validation accuracy and loss graphs automatically at the end of model training.
For the testing implementation, we fixed a previous bug with image padding for different patch sizes, added a new 4D image padding function, and replaced the slice-by-slice format previously used to generate model predictions on unseen images with a new method that uses a moving 3D window applied serially across the image volume. This allows higher resolution images to fit into available GPU memory. In addition, the method now offers support for 3 different trainable network acrchitechures:
1. [3D Unet](https://arxiv.org/abs/1606.06650)
2. [Feature pyramid network-ResNet50 (with bottleneck ResNet modules)](https://arxiv.org/abs/1612.03144)
3. [Panoptic feature pyramid network-ResNet50 (with preactivated ResNet modules)](https://arxiv.org/abs/1901.02446)

## Prerequisites before running JCnet

### Pre-processing:
A few standard MRI preprocessing steps are necessary before training or testing a JCnet model:
1. Bias field correction - can use either [N4 bias correction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855/) or [MICO](https://www.sciencedirect.com/science/article/abs/pii/S0730725X14000927).
2. Skull-stripping - we recommend using the [MONSTR skull-stripping algorithm](https://pubmed.ncbi.nlm.nih.gov/27864083/) in PML cases, which is publicly available and can be found [here](https://www.nitrc.org/projects/monstr).
3. Transformation to the standard MNI-ICBM 152 atlas space, which is available for download [here](http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009).
4. Co-registration of different MRI channels or contrasts (i.e. T1-weighted, fluid-attenuated inversion recovery, T2-weighted, and proton density images).

### Hardware Requirements:
1. Operating System: Linux.
2. CPU Number/Speed: we recommend using a processor with at least 8 cores, 2GHz speed, and multithreading capability.
3. RAM: 64+GB recommended (depending on the size of the training dataset and maximum number of training patches per subject).
4. GPU: recommend a dedicated graphics card with at least 8GB of VRAM (ex. NVIDIA RTX 2080 Ti, Titan X, or v100 models). If our current pre-trained models do not fit into GPU memory during testing, we recommend downscaling the network parameters (base filters, patch size, or architechure in this order). These models can be provided upon request.

### Software Requirements:
1. Python v3.6
2. Keras v2.2.4
3. Tensorflow GPU version v1.13+ (TF v2 is not currently supported)
4. Several open source python packages, please see [requirements.txt](https://github.com/omarallouz/JCnet/blob/master/requirements.txt)
To install python dependency packages, you can point your pip manager using the terminal to the text file as follows:
```
pip3 install -r requirements.txt 
```
### Training call examples
```
# Brain Extraction training:
python JCnet_BrainExtraction_Train.py --atlasdir /path/to/atlas/dir/ --natlas 31 --psize 64 64 64 --maxpatch 1000 --batchsize 8 --basefilters 32 --modalities T1 FL T2 PD --epoch 50 --outdir /path/to/output/dir/to/save/models/ --save 1 --gpuids 0 1 2 3 --loss focal --model FPN
```
```
# Lesion Segmentation training:
python JCnet_LesionSeg_Train.py --atlasdir /path/to/atlas/dir/ --natlas 31 --psize 64 64 64 --maxpatch 1000 --batchsize 8 --basefilters 32 --modalities T1 FL T2 PD --epoch 50 --outdir /path/to/output/dir/to/save/models/ --save 1 --gpuids 0 1 2 3 --loss focal --model FPN
```
### Testing call examples
```
# Brain Extraction testing:
python JCnet_BrainExtraction_Test.py --models /path/to/model/files/containing/in/\*Orient012\*.h5 /path/to/model/files/ending/in/\*Orient120\*.h5 /path/to/model/files/ending/in/\*Orient201\*.h5  --images /path/to/T1/niftifile/\*.nii.gz /path/to/FL/niftifile/\*.nii.gz /path/to/T2/niftifile/\*.nii.gz /path/to/PD/niftifile/\*.nii.gz --modalities T1 FL T2 PD --psize 64 64 64 --outdir /path/to/output/dir/to/save/results/ --threshold 0.5
```
```
# Lesion Segmentation testing:
python JCnet_LesionSeg_Test.py --models /path/to/model/files/containing/in/\*Orient012\*.h5 /path/to/model/files/ending/in/\*Orient120\*.h5 /path/to/model/files/ending/in/\*Orient201\*.h5  --images /path/to/T1/niftifile/\*.nii.gz /path/to/FL/niftifile/\*.nii.gz /path/to/T2/niftifile/\*.nii.gz /path/to/PD/niftifile/\*.nii.gz --modalities T1 FL T2 PD --psize 64 64 64 --outdir /path/to/output/dir/to/save/results/ --threshold 0.35
```