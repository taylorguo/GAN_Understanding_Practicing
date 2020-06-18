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
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name="noise_input")
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# A boolean to indicate BN if training or inference
is_training = tf.placeholder(tf.bool)

model_dir = "Model"

# No this function below tensorflow-1.12
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

    ## 1. ## 构建生成器和判别器损失函数
    # 构建生成器
    gen_sample = generator(noise_input)
    # 构建判别器: 1. 对真实图片判别;  2. 对生成图片判别
    discriminator_real = discriminator(real_image_input)
    discriminator_fake = discriminator(gen_sample, reuse=True)
    # 真实图像: 标签1; 生成图像: 标签0
    discriminator_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = discriminator_real, labels = tf.ones([batch_size], dtype=tf.int32)))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.zeros([batch_size], dtype=tf.int32)))
    # 判别器总损失函数
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    # 生成器损失函数: 生成器生成的图像, 希望这个图像尽可能地与真实图像一致, 也就是判别器认为生成器生成图像为1时, 任务成功, 所以标签为1 
    generator_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.ones([batch_size], dtype = tf.int32)))
    ## 2. ## 创建优化器
    optimizer_generator = tf.train.AdamOptimizer(learning_rate= lr_generator, beta1=0.5, beta2= 0.999)
    optimizer_discriminator = tf.train.AdamOptimizer(learning_rate= lr_discriminator, beta1= 0.5, beta2=0.999)
    ## 3. ## 构建训练变量和训练操作
    # 获取所有变量用于优化器更新
    generator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")
    discriminator_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator")
    # BN operation in training
    # BN中的每个batch的均值和方差并不是训练得到的,而是直接计算后更新的; 放在collection — tf.GraphKeys.UPDATE_OPS中, 每轮迭代前需要插入该操作
    generator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Generator")
    with tf.control_dependencies(generator_update_ops):
        train_generator = optimizer_generator.minimize(generator_loss, var_list=generator_var)
    discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="Discriminator")
    with tf.control_dependencies(discriminator_update_ops):
        train_discriminator = optimizer_discriminator.minimize(discriminator_loss, var_list=discriminator_var)
    ## 4. ## 初始化全局变量
    init = tf.global_variables_initializer()

    ## 5. ## 开始训练
    saver = tf.train.Saver()
    model_name = model_dir + "/dcgan.ckpt"
    sess = tf.Session()
    sess.run(init)
    # Training
    for i in range(1, num_steps + 1):
        # 准备输入数据, 不需要标签
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        # Rescale to [-1, 1] for input of discriminator
        batch_x = batch_x * 2. - 1.

        # Discriminator Training
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, discriminator_learning = sess.run([train_discriminator, discriminator_loss], feed_dict={real_image_input: batch_x, noise_input:z, is_training:True})

        # Generator Training
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
        _, generator_learning = sess.run([train_generator, generator_loss], feed_dict={noise_input: z, is_training:True})

        if i % 500 == 0 or i == 1:
            print("Step %d: Generator Loss: %f, Discriminator Loss: %f" % (i, generator_learning, discriminator_learning))
            saver.save(sess, model_name)

def predict():

    meta_file = "/dcgan.ckpt.meta"
    saver = tf.train.import_meta_graph(model_dir + meta_file)
    graph = tf.get_default_graph()
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    # for i in tensor_name_list:
    #     print(i)

    x = graph.get_tensor_by_name("Placeholder:0")
    y = graph.get_tensor_by_name()


    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        print(" *** Model Loaded! ***")

        gen_sample = generator(noise_input)

        n = 6
        canvas = np.empty((28 * n, 28 * n))

        for i in range(n):
            z = np.random.uniform(-1., 1., size=[n, noise_dim])

            g = sess.run(gen_sample, feed_dict={})

if __name__ == '__main__':
    # train()
    predict()