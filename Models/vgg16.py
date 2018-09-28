from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Input
from keras import layers
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import MaxPooling2D, Cropping2D
from keras.layers import BatchNormalization
from keras.models import Model, optimizers
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import parameters as para

from keras.datasets import cifar100
import cv2, os
import numpy as np
from keras.utils import np_utils
from keras.callbacks import *
from utils import memory_usage


def vgg_block(input_tensor, filters=64, group="0", _layers=2, dilation_rate=(1, 1), kernel_size=(3, 3), strides=(2, 2),
              maxpool=(2, 2), activation="relu"):
    conv_base_name = "vgg_Group_" + group + "_Conv_"
    pool_name = "vgg_Group_" + group + "_Pool"
    x=input_tensor
    for block in range(_layers):
        conv_name = conv_base_name + str(block)
        x = Conv2D(filters,
                   kernel_size,
                   activation=activation,
                   padding='same',
                   name=conv_name,
                   dilation_rate=dilation_rate
                   )(x)
    x = MaxPooling2D(maxpool, strides=strides, name=pool_name)(x)

    return x


def VGG16(include_top=False, input_tensor=None, input_shape=(para.img_rows, para.img_cols, para.channels)):
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    #   Group 1
    x = vgg_block(img_input, filters=64, group="1", _layers=2, dilation_rate=(1, 1), kernel_size=(3, 3), strides=(2, 2),
                  maxpool=(2, 2), activation="relu")
    #   Group 2
    x = vgg_block(x, filters=128, group="2", _layers=2, dilation_rate=(1, 1), kernel_size=(3, 3), strides=(2, 2),
                  maxpool=(2, 2), activation="relu")
    #   Group 3
    x = vgg_block(x, filters=256, group="3", _layers=3, dilation_rate=(1, 1), kernel_size=(3, 3), strides=(2, 2),
                  maxpool=(2, 2), activation="relu")
    #   Group 4
    x = vgg_block(x, filters=512, group="4", _layers=3, dilation_rate=(1, 1), kernel_size=(3, 3), strides=(1, 1),
                  maxpool=(1, 1), activation="relu")
    #   Group 5
    x = vgg_block(x, filters=512, group="5", _layers=3, dilation_rate=(1, 1), kernel_size=(3, 3), strides=(1, 1),
                  maxpool=(1, 1), activation="relu")

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg-16')

    return model


if __name__ == '__main__':
    model = VGG16()
    print(model.summary())