import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import os
import sys
from keras.layers import Lambda
from keras.layers import BatchNormalization
K.set_image_data_format('channels_last')
path = os.path.dirname(sys.argv[0])
path=os.path.abspath(path)
sys.path.append(path)

# ResNet design methodology adopted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py

def identity_block2D(input_tensor, kernel_size, filters, use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    nb_filter1 = nb_filter2 = nb_filter3 = filters

    x = KL.Conv2D(nb_filter1, (1, 1), padding='same', strides=(1,1), use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', strides=(1,1), use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), padding='same', strides=(1,1), use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=-1)(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def identity_block3D(input_tensor, kernel_size, filters, use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    nb_filter1 = nb_filter2 = nb_filter3 = filters

    x = KL.BatchNormalization(axis=-1)(input_tensor)
    x = KL.Activation('relu')(x)
    x = KL.Conv3D(nb_filter1, (1, 1, 1), padding='same', strides=(1,1,1), use_bias=use_bias)(x)

    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', strides=(1,1,1), use_bias=use_bias)(x)

    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Conv3D(nb_filter3, (1, 1, 1), padding='same', strides=(1,1,1), use_bias=use_bias)(x)

    x = KL.Add()([x, input_tensor])
    return x

def conv_block2D(input_tensor, kernel_size, filters, strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1 = nb_filter2 = nb_filter3 = filters

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides, use_bias=use_bias, padding='same')(input_tensor)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=-1)(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,padding='same', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(axis=-1)(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu')(x)
    return x


def conv_block3D(input_tensor, kernel_size, filters, strides=(2, 2, 2), use_bias=True, train_bn=True):


    nb_filter1 = nb_filter2 = nb_filter3 = filters

    x = KL.BatchNormalization(axis=-1)(input_tensor)
    x = KL.Activation('relu')(x)
    x = KL.Conv3D(nb_filter1, (1, 1, 1), strides=strides, use_bias=use_bias, padding='same')(x)

    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Activation('relu')(x)
    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', use_bias=use_bias)(x)

    x = KL.BatchNormalization(axis=-1)(x)
    x = KL.Conv3D(nb_filter3, (1, 1, 1), use_bias=use_bias)(x)

    shortcut = KL.BatchNormalization(axis=-1)(input_tensor)
    shortcut = KL.Activation('relu')(shortcut)
    shortcut = KL.Conv3D(nb_filter3, (1, 1, 1), strides=strides,padding='same', use_bias=use_bias)(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu')(x)
    return x


def resnet_graph2D(input_image, basefilters, architecture, stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """

    assert architecture in ["resnet50", "resnet101"]
    x = KL.Conv2D(basefilters, (3, 3), strides=(1, 1), use_bias=True, padding='same')(input_image)
    x = KL.BatchNormalization(axis=-1)(x)

    C1 = KL.Activation('relu')(x)
    x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(C1)

    # Stage 2, x is 64x64 patches
    x = conv_block2D(x, 3, 2*basefilters, strides=(1, 1), train_bn=train_bn)
    C2 = x = identity_block2D(x, 3, 2*basefilters, train_bn=train_bn)

    # Stage 3, 64x64 patches
    x = conv_block2D(x, 3, basefilters*4, strides=(2,2), train_bn=train_bn)
    C3 = x = identity_block2D(x, 3, basefilters*4, train_bn=train_bn)

    # Stage 4, 32x32 patches
    x = conv_block2D(x, 3, basefilters*8, strides=(2,2), train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]

    # x has 16x16 patches
    for i in range(block_count):
        x = identity_block2D(x, 3, basefilters*8, train_bn=train_bn)
    C4 = x = identity_block2D(x, 3, basefilters*8, train_bn=train_bn)

    # Stage 5, C4 has 16x16 patches
    # Ideally for 3D patches, max we can use is 32x32x32 which makes stage5 negligible, so for 3D, stage5 will be excluded
    if stage5:
        x = conv_block2D(x, 3, basefilters*16, strides=(2,2), train_bn=train_bn)
        C5 = x = identity_block2D(x, 3, basefilters*16, train_bn=train_bn)
        # C5 has 8x8 patches
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def resnet_graph3D(input_image, basefilters, architecture, stage5=False, train_bn=True):

    assert architecture in ["resnet50", "resnet101"]
    # Stage 1, input is 64x64x64 patches

    x = KL.Conv3D(basefilters, (3, 3, 3), strides=(1, 1, 1), use_bias=True, padding='same')(input_image)
    x = KL.BatchNormalization(axis=-1)(x)
    C1 = KL.Activation('relu')(x)

    # Stage 2, x is 32x32x32 patches
    x = KL.MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(C1)
    x = conv_block3D(x, 3, 2*basefilters, strides=(1, 1, 1), train_bn=train_bn)
    C2 = x = identity_block3D(x, 3, 2*basefilters, train_bn=train_bn)


    # Stage 3, 16x16x16 patches
    x = conv_block3D(x, 3, basefilters*4, strides=(2,2,2), train_bn=train_bn)
    C3 = x = identity_block3D(x, 3, basefilters*4, train_bn=train_bn)


    # Stage 4, 8x8x8 patches
    x = conv_block3D(x, 3, basefilters*8, strides=(2,2,2), train_bn=train_bn)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    # x has 8x8x8 patches
    for i in range(block_count):
        x = identity_block3D(x, 3, basefilters*8, train_bn=train_bn)
    x = identity_block3D(x, 3, basefilters * 8, train_bn=train_bn)
    x = KL.BatchNormalization(axis=-1)(x)
    C4 = x = KL.Activation('relu')(x)

    # Stage 5, C4 has 8x8x8 patches
    # Ideally for 3D patches, max we can use is 32x32x32 which makes stage5 negligible,
    # so for 3D, stage5 should be excluded
    if stage5:
        x = conv_block3D(x, 3, basefilters*16, strides=(2,2,2), train_bn=train_bn)
        C5 = x = identity_block3D(x, 3, basefilters*16, train_bn=train_bn)
        # C5 has 4x4x4 patches
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]



def FPN2D(input, basefilter):
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    C1, C2, C3, C4, C5 = resnet_graph2D(input, basefilter, 'resnet50', stage5=True, train_bn=TRAIN_BN)
    # Top-down Layers
    TOP_DOWN_FILTERS=basefilter*4  # Default basefilter is 64 (i.e for C1), and config.TOP_DOWN_PYRAMID_SIZE=256
    P5 = KL.Conv2D(TOP_DOWN_FILTERS, (1, 1),  padding='same', strides=(1,1), activation='relu')(C5)
    P4 = KL.Add()([KL.UpSampling2D(size=(2, 2))(P5),
                   KL.Conv2D(TOP_DOWN_FILTERS, (1, 1), padding='same', strides=(1,1), activation='relu')(C4)])
    P3 = KL.Add()([KL.UpSampling2D(size=(2, 2))(P4),
                   KL.Conv2D(TOP_DOWN_FILTERS, (1, 1), padding='same', strides=(1,1), activation='relu')(C3)])
    P2 = KL.Add()([KL.UpSampling2D(size=(2, 2))(P3),
                   KL.Conv2D(TOP_DOWN_FILTERS, (1, 1), padding='same', strides=(1,1), activation='relu')(C2)])
    P1 = KL.Add()([KL.UpSampling2D(size=(2, 2))(P2),
                   KL.Conv2D(TOP_DOWN_FILTERS, (1, 1), padding='same', strides=(1, 1), activation='relu')(C1)])


    # Attach 3x3 conv to all P layers to get the final feature maps.
    # Not sure why
    P1 = KL.Conv2D(TOP_DOWN_FILTERS, (3, 3), padding="SAME", strides=(1,1), activation='relu')(P1)
    P2 = KL.Conv2D(TOP_DOWN_FILTERS, (3, 3), padding="SAME", strides=(1,1), activation='relu')(P2)
    P3 = KL.Conv2D(TOP_DOWN_FILTERS, (3, 3), padding="SAME", strides=(1,1), activation='relu')(P3)
    P4 = KL.Conv2D(TOP_DOWN_FILTERS, (3, 3), padding="SAME", strides=(1,1), activation='relu')(P4)
    P5 = KL.Conv2D(TOP_DOWN_FILTERS, (3, 3), padding="SAME", strides=(1,1), activation='relu')(P5)


    return [P1,P2,P3,P4,P5]


def FPN3D(input, basefilter):
    TRAIN_BN = False  # Defaulting to False since batch size is often small
    stage5 = False
    C1, C2, C3, C4, _ = resnet_graph3D(input, basefilter, 'resnet50', stage5=stage5, train_bn=TRAIN_BN)
    # Top-down Layers
    TOP_DOWN_FILTERS=basefilter*4  # Default basefilter is 64 (i.e for C1), and config.TOP_DOWN_PYRAMID_SIZE=256
    if stage5==True:
        P5 = KL.Conv3D(TOP_DOWN_FILTERS, (1, 1, 1),  padding='same', strides=(1,1,1), activation='relu')(C5)
        P4 = KL.Add()([KL.UpSampling3D(size=(2, 2, 2))(P5),
                       KL.Conv3D(TOP_DOWN_FILTERS, (1, 1, 1), padding='same', strides=(1, 1, 1), activation='relu')(
                           C4)])
    else:
        P5 = None
        P4 = KL.Conv3D(TOP_DOWN_FILTERS, (1, 1, 1), padding='same', strides=(1, 1, 1), activation='relu')(C4)

    P3 = KL.Add()([KL.UpSampling3D(size=(2, 2, 2))(P4),
                   KL.Conv3D(TOP_DOWN_FILTERS, (1, 1, 1), padding='same', strides=(1,1,1), activation='relu')(C3)])
    P2 = KL.Add()([KL.UpSampling3D(size=(2, 2, 2))(P3),
                   KL.Conv3D(TOP_DOWN_FILTERS, (1, 1, 1), padding='same', strides=(1,1,1), activation='relu')(C2)])
    P1 = KL.Add()([KL.UpSampling3D(size=(2, 2, 2))(P2),
                   KL.Conv3D(TOP_DOWN_FILTERS, (1, 1, 1), padding='same', strides=(1, 1, 1), activation='relu')(C1)])


    # Attach 3x3 conv to all P layers to get the final feature maps.
    # Not sure why
    P1 = KL.Conv3D(TOP_DOWN_FILTERS, (3, 3, 3), padding="SAME", strides=(1,1,1), activation='relu')(P1)
    P2 = KL.Conv3D(TOP_DOWN_FILTERS, (3, 3, 3), padding="SAME", strides=(1,1,1), activation='relu')(P2)
    P3 = KL.Conv3D(TOP_DOWN_FILTERS, (3, 3, 3), padding="SAME", strides=(1,1,1), activation='relu')(P3)
    P4 = KL.Conv3D(TOP_DOWN_FILTERS, (3, 3, 3), padding="SAME", strides=(1,1,1), activation='relu')(P4)
    if stage5:
        P5 = KL.Conv3D(TOP_DOWN_FILTERS, (3, 3, 3), padding="SAME", strides=(1,1,1), activation='relu')(P5)
    else:
        P5 = None


    return [P1,P2,P3,P4,P5]