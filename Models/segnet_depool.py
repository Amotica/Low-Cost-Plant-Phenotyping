from __future__ import absolute_import
from __future__ import print_function
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import parameters as para
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.backend import gradients, sum, repeat_elements


class DePool2D_Index(UpSampling2D):
    """https://github.com/nanopony/keras-convautoencoder/blob/c8172766f968c8afc81382b5e24fd4b57d8ebe71/autoencoder_layers.py#L24
    Simplar to UpSample, yet traverse only maxpooled elements."""
    input_ndim = 4

    def __init__(self, pool2d_layer, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = repeat_elements(X, self.size[0], axis=2)
            output = repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = repeat_elements(X, self.size[0], axis=1)
            output = repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = gradients(sum(self._pool2d_layer.get_output(train)), self._pool2d_layer.get_input(train)) * output

        return f


def SegNet(input_shape=(para.img_cols, para.img_rows, para.channels), classes=para.num_classes, input_tensor=None):
    #img_input = Input(shape=input_shape)

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=96,
                                      data_format=K.image_data_format(),
                                      include_top=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    #print(img_input.shape)
    x = img_input
    # Encoder
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(128, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    encoder = Model(img_input, x, name='vgg16_encoder')
    encoder_layers = [layer for _, layer in enumerate(encoder.layers)]
    encoder_layers.reverse()

    #for i, layer in enumerate(encoder_layers):
        #print(i, layer)

    # Decoder
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DePool2D_Index(encoder_layers[9], size=encoder_layers[9].pool_size, input_shape=encoder.output_shape[1:])(x)
    x = Conv2D(encoder_layers[12].filters, encoder_layers[12].kernel_size, padding=encoder_layers[12].padding)(x)
    #x = UpSampling2D(size=(2, 2))(x)
    #x = Conv2D(256, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = Conv2D(256, (3, 3), padding="same")(x)
    x = Conv2D(encoder_layers[15].filters, encoder_layers[15].kernel_size, padding=encoder_layers[15].padding)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    #x = Conv2D(256, (3, 3), padding="same")(x)
    x = Conv2D(encoder_layers[18].filters, encoder_layers[18].kernel_size, padding=encoder_layers[18].padding)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DePool2D_Index(encoder_layers[19], size=encoder_layers[19].pool_size, input_shape=encoder.output_shape[1:])(x)
    x = Conv2D(encoder_layers[22].filters, encoder_layers[22].kernel_size, padding=encoder_layers[22].padding)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(encoder_layers[25].filters, encoder_layers[25].kernel_size, padding=encoder_layers[25].padding)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DePool2D_Index(encoder_layers[26], size=encoder_layers[26].pool_size, input_shape=encoder.output_shape[1:])(x)
    x = Conv2D(encoder_layers[29].filters, encoder_layers[29].kernel_size, padding=encoder_layers[29].padding)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(encoder_layers[32].filters, encoder_layers[32].kernel_size, padding=encoder_layers[32].padding)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    model = Model(img_input, x)
    cols = model.output_shape[1]
    rows = model.output_shape[2]
    x = Conv2D(classes, (1, 1), padding="valid")(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x) #  *****************
    x = Activation("softmax")(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x)
    return model, rows, cols


if __name__ == '__main__':
    model, rows, cols = SegNet(input_shape=(96, 320, 3))
    print(rows, cols)
    print(model.summary())
