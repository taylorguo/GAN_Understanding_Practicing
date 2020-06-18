import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Generator:
    def __init__(self):
        self.name = "Generator"
        self.size = 64/16
        self.channel = 3
        self.channel_list = [1024, 512, 256, 128, 3]
    
    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            x = tf.layers.dense(z, self.size * self.size * self.channel_list[0], activation=tf.nn.relu())


class DCGAN():
    def __init__(self, generator, discriminator, data):
        self.genenrator = generator
        self.discriminator = discriminator
        self.data = data