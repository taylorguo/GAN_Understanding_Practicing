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
    # C64
    d = Conv2D(64, (4, 4), strides=(2,2), padding="same", kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    patch_out = Activation("sigmoid")(d)
    # define Model
    model = Model([src_image, target_image], patch_out)
    model.summary()
    # compile model
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    return model

def encoder_block(layer_in, n_filers, batchnorm=True):
    # weight initialization
    init = keras.initializers.RandomNormal(stddev=0.02)
    # add downsampleing layer
    g = Conv2D(n_filers, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training = True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

def decoder_block(layer_in, skip_in, n_filers, dropout=True):
    init = keras.initializers.RandomNormal(stddev=0.02)