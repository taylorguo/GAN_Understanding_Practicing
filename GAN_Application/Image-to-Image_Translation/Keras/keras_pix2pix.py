import numpy as np
import keras
from keras import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation
from keras.layers import Concatenate, Dropout, BatchNormalization


def define_discriminator(image_shape):
    # Weight Initialize
    init = keras.initializers.RandomNormal(stddev=0.02)

    # source image input
    src_image = Input(shape=image_shape)
    # target image input 
    target_image = Input(shape=image_shape)
    # concatenate images channel-wise
    merged = Concatenate()[src_image, target_image]

    # build D net
    d = Conv2D(64, (4,4), strides=(2,2), padding="same", kernel_initializer=init)
    d = LeakyReLU(alpha=0.2)(d)


