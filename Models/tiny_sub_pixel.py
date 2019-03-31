from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import parameters as para
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.engine.topology import get_source_inputs
from Models import subPixelConv2D


def tinySubPixelModel(input_shape=(para.img_cols, para.img_rows, para.channels), classes=para.num_classes, input_tensor=None,
           dilation_rate=(1, 1)):

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=96,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input

    # Encoder
    x = SeparableConv2D(64, (3, 3), padding="same", name="block1_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(64, (3, 3), padding="same", name="block1_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(128, (3, 3), padding="same", name="block2_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(128, (3, 3), padding="same", name="block2_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(256, (3, 3), padding="same", name="block3_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(256, (3, 3), padding="same", name="block3_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(256, (3, 3), padding="same", name="block3_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = SeparableConv2D(512, (3, 3), padding="same", name="block4_conv0")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(512, (3, 3), padding="same", name="block4_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SeparableConv2D(512, (3, 3), padding="same", name="block4_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    #   =============
    #   The Sub-Pixel Block Replacing the Decoder....
    #   =============
    scale = 8
    o = subPixelConv2D.SubpixelConv2D(input_shape, scale=scale)(x)

    #   =============
    #   The Sub-Pixel Block Replacing the Decoder....
    #   =============
    scale = 8
    o = subPixelConv2D.SubpixelConv2D(input_shape, scale=scale)(x)

    model = Model(img_input, o)
    cols = model.output_shape[1]
    rows = model.output_shape[2]

    o = Conv2D(classes, (1, 1), padding="valid")(o)
    o = Reshape((cols * rows, classes))(o)  # *****************
    o = Activation("softmax")(o)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, o)
    return model, rows, cols


if __name__ == '__main__':
    model, rows, cols = tinySubPixelModel()
    print(rows, cols)
    print(model.summary())
