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
    g = Conv2DTranspose(n_filers, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training = True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()[g, skip_in]
    g = Activation("relu")(g)
    return g

def define_generator(image_shape=(256, 256, 3)):
    init = keras.initializers.RandomNormal(stddev=0.02)

    in_image = Input(shape=image_shape)
    # encoder
    e1 = encoder_block(in_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e1, 256)
    e4 = encoder_block(e1, 512)
    e5 = encoder_block(e1, 512)
    e6 = encoder_block(e1, 512)
    e7 = encoder_block(e1, 512)
    # Bottleneck, no BN and ReLU
    b = Conv2D(512, (4, 4), strides=(2,2), padding="same", kernel_initializer=init)(e7)
    b = Activation("relu")(b)
    # decoder
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d7)
    out_image = Activation("tanh")(g)
    # define model
    model = Model(in_image, out_image)
    model.summary()
    return model

def define_gan(g, d, image_shape):
    d.trainable = False
    src_image = Input(shape=image_shape)
    gen_out = g(src_image)
    dis_out = d([src_image, gen_out])
    # src image as input, generated image and classification output
    model = Model(src_image, [dis_out, gen_out])
    model.summary()
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]) 
    return model