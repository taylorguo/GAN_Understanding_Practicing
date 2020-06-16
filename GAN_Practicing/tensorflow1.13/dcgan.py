import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# training parameters
num_steps = 10000
batch_size = 128
lr_generator = 0.002
lr_discriminator = 0.002

# network parameters
image_dim = 784
noise_dim = 100

# build network
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# A boolean to indicate BN if training or inference
is_training = tf.placeholder(tf.bool)

def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

# define generator
# Input: noise; Output: image
# Only Use BN in training
def generator(x, reuse = False):
    with tf.variable_scope("Generator", reuse=reuse):
        # 第一层是全连接层, 神经元个数是7*7*128个, 输入是噪声batch*100
        x = tf.layers.dense(x, units= 7 * 7 * 128)
        x = tf.layers.batch_normalization(x, training = is_training)
        x = tf.nn.relu(x)
        # reshape成4维张量: (batch, height, width, channel), 这里是(batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        # 反卷积层1: 卷积核 5*5*128, 64个, 步长为2 (该函数参数: input, filters数量, kernel_size, strides, padding)
        # 输入张量: (batch, 7, 7, 128); 输出张量: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding="same")
        # BN: 在channel上做normalize
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # 反卷积层2: 卷积核 5*5*128, 1个, 步长为2 (该函数参数: input, filters数量, kernel_size, strides, padding)
        # 输入张量: (batch, 14, 14, 64); 输出张量: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding="same")
        x = tf.nn.tanh(x)
        return x

# define discriminator
# 输入: 图像, 输出: 预测结果(Real / Fake Image)
# 训练时才用BN
def discriminator(x, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        # 卷积层1: 输入x, 卷积核大小 5x5, 64个, 步长2
        x = tf.layers.conv2d(x, 64, 5, strides=2, padding="same")
        x = tf.layers.batch_normalization(x, training=is_training)
        # x = leakyrelu(x)
        x = tf.nn.leaky_relu(x)
        # 卷积层2: 输入x, 卷积核大小 5x5, 128个, 步长2
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding="same")
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        #
        x = tf.reshape(x, shape=[-1, 7*7*128])
        #
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = tf.layers.dense(x, 2)
        return x

def train():
    # 构建生成器
    gen_sample = generator(noise_input)
    # 构建判别器: 1. 对真实图片判别;  2. 对生成图片判别
    discriminator_real = discriminator(real_image_input)
    discriminator_fake = discriminator(gen_sample, reuse=True)
    # 真实图像: 标签1; 生成图像: 标签0
    discriminator_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = discriminator_real, labels = tf.ones([batch_size], dtype=tf.int32)))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.zeros([batch_size], dtype=tf.int32)))
