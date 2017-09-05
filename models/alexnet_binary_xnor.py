import tensorflow as tf
from tf_gemm_op import xnor_gemm
import numpy as ny

BN_EPSILON = 1e-5

@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

# Alexnet with xnor for cifar10
class AlexBinaryNet:

#(x, keep_prob, n_classes=n_classes, imagesize, img_channel)
    def __init__(self, binary, x, keep_prob, n_classes, imagesize, img_channel, batch_norm, phase):
        self.binary = binary
        self.n_classes = n_classes
        self.keep_prob = keep_prob
        self.input = x
        self.G = tf.get_default_graph()
        self.conv_layers(batch_norm, phase)
        self.imagesize = imagesize
        self.img_channel = img_channel


    def hard_sigmoid(self, x):
        return tf.clip_by_value((x + 1.) / 2, 0, 1)

    def binary_tanh_unit(self, x):
        return 2 * self.hard_sigmoid(x) - 1

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)
            # E = tf.reduce_mean(tf.abs(x))
            # return tf.sign(x) * E

    def quantize_filter(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            # Wb, alpha
            return tf.sign(x), tf.reduce_mean(tf.abs(x), [0, 1, 2])


    def weight_variable(self, name, shape):
        init=tf.get_variable(name,shape,initializer=tf.contrib.layers.xavier_initializer())
        return init

    def bias_variable(self, name, shape):
        init = tf.get_variable(name,shape,initializer= tf.zeros_initializer() )
        return init

    def conv2d(self, x, W, s):
        return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

    def max_pool(self, l_input, k, s):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1,s,s,1], padding='VALID')

    def conv2d_1x1(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')

    def conv_layers(self, batch_norm, phase):

        if self.binary:
            # based on the paper, all the bias numbers have been deleted.
            with tf.name_scope('conv1_bin') as scope:
                W_conv1 = self.weight_variable('w1',shape=[3, 3, 3, 64])
                h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input, W_conv1, strides=[1, 1, 1, 1], padding='VALID'))
                h_pool1 = self.max_pool(h_conv1, k=3, s=2)
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_pool1, [0], keep_dims=True)
                    h_pool1 = tf.nn.batch_normalization(
                        h_pool1, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv2_bin') as scope:
                W_conv2 = self.weight_variable(name='w2', shape=[3, 3, 64, 192])
                Wb_conv2, alpha_2 = self.quantize_filter(W_conv2)
                ##Wb_conv2 = W_conv2
                #h_conv2 = tf.nn.relu(self.conv2d(h_pool1, Wb_conv2, s=1))
                h_conv2 = self.conv2d(tf.nn.relu(h_pool1), Wb_conv2, s=1)
                h_pool2 = self.max_pool(h_conv2, k=3, s=2)
                h_pool2_bin = self.quantize(h_pool2)
                ##h_pool2_bin = h_pool2
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_pool2_bin, [0], keep_dims=True)
                    h_pool2_bin = tf.nn.batch_normalization(
                        h_pool2_bin, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv3_bin') as scope:
                W_conv3 = self.weight_variable(name='w3',shape=[3, 3, 192, 384])
                Wb_conv3, alpha_3 = self.quantize_filter(W_conv3)
                #Wb_conv3 = W_conv3
                #h_conv3 = tf.nn.relu(self.conv2d(h_pool2_bin, Wb_conv3, s=1))
                h_conv3 = self.conv2d(tf.nn.relu(h_pool2_bin), Wb_conv3, s=1)
                h_conv3_bin = self.quantize(h_conv3)
                #h_conv3_bin = h_conv3
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_conv3_bin, [0], keep_dims=True)
                    h_conv3_bin = tf.nn.batch_normalization(
                        h_conv3_bin, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv4_bin') as scope:
                W_conv4 = self.weight_variable(name='w4',shape=[3,3,384,384])
                Wb_conv4, alpha_4 = self.quantize_filter(W_conv4)
                #Wb_conv4 = W_conv4
                #h_conv4 = tf.nn.relu(self.conv2d(h_conv3_bin, Wb_conv4,s=1))
                h_conv4 = self.conv2d(tf.nn.relu(h_conv3_bin), Wb_conv4, s=1)
                h_conv4_bin = self.quantize(h_conv4)
                #h_conv4_bin = h_conv4
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_conv4_bin, [0], keep_dims=True)
                    h_conv4_bin = tf.nn.batch_normalization(
                        h_conv4_bin, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv5_bin') as scope:
                W_conv5 = self.weight_variable(name='w5',shape=[3,3,384,256])
                Wb_conv5, alpha_5 = self.quantize_filter(W_conv5)
                #Wb_conv5 = W_conv5
                #h_conv5 = tf.nn.relu(self.conv2d(h_conv4_bin, Wb_conv5, s=1))
                h_conv5 = self.conv2d(tf.nn.relu(h_conv4_bin), Wb_conv5, s=1)
                h_pool5 = self.max_pool(h_conv5, k=3, s=2)
                h_pool5_bin = self.quantize(h_pool5)
                #h_pool5_bin = h_pool5
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_pool5_bin, [0], keep_dims=True)
                    h_pool5_bin = tf.nn.batch_normalization(
                        h_pool5_bin, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('fc6_bin') as scope:
                fcw_init = tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
                W_fc6 = tf.get_variable(name='fc6_w',shape=[2, 2, 256, 4096], initializer=fcw_init)
                Wb_fc6 = self.quantize(W_fc6)
                #Wb_fc6 = W_fc6
                #h_fc6 = tf.nn.relu(tf.nn.conv2d(h_pool5_bin, Wb_fc6, strides=[1, 1, 1, 1], padding='VALID'))
                h_fc6 = tf.nn.conv2d(tf.nn.relu(h_pool5_bin), Wb_fc6, strides=[1, 1, 1, 1], padding='VALID')
                h_fc6_d = tf.nn.dropout(h_fc6, keep_prob=self.keep_prob)
                h_fc6_bin = self.quantize(h_fc6_d)
                #h_fc6_bin = h_fc6_d
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_fc6_bin, [0], keep_dims=True)
                    h_fc6_bin = tf.nn.batch_normalization(
                        h_fc6_bin, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('fc7_bin') as scope:
                W_fc7=tf.get_variable(name='fc7_w',shape=[1, 1, 4096, 4096],initializer=fcw_init)
                Wb_fc7 = self.quantize(W_fc7)
                #Wb_fc7 = W_fc7
                #h_fc7 = tf.nn.relu(self.conv2d(h_fc6_bin, Wb_fc7, s=1))
                h_fc7 = self.conv2d(tf.nn.relu(h_fc6_bin), Wb_fc7, s=1)
                h_fc7_d = tf.nn.dropout(h_fc7, keep_prob=self.keep_prob)
                h_fc7_bin = self.quantize(h_fc7_d)
                #h_fc7_bin = h_fc7_d
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_fc7_bin, [0], keep_dims=True)
                    h_fc7_bin = tf.nn.batch_normalization(
                        h_fc7_bin, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('fcout_bin') as scope:
                fcout_init = tf.zeros_initializer()
                W_fc8 = tf.get_variable(name='fc8_w',shape=[1, 1, 4096, self.n_classes], initializer=fcout_init)
                Wb_fc8 = self.quantize(W_fc8)
                #Wb_fc8 = W_fc8
                h_fc8 = self.conv2d(h_fc7_bin, Wb_fc8, s=1)
                self.output = tf.squeeze(h_fc8, [1, 2], name='fcout/squeezed')
        else:
            ## Alexnet v2 without binary operations
            with tf.name_scope('conv1') as scope:
                W_conv1 = self.weight_variable('w1',shape=[3, 3, 3, 64])
                h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input, W_conv1, strides=[1, 1, 1, 1], padding='VALID'))
                h_pool1 = self.max_pool(h_conv1, k=3, s=2)
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_pool1, [0], keep_dims=True)
                    h_pool1 = tf.nn.batch_normalization(
                        h_pool1, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv2') as scope:
                W_conv2 = self.weight_variable(name='w2', shape=[3, 3, 64, 192])
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, s=1))
                h_pool2 = self.max_pool(h_conv2, k=3, s=2)
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_pool2, [0], keep_dims=True)
                    h_pool2 = tf.nn.batch_normalization(
                        h_pool2, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv3') as scope:
                W_conv3 = self.weight_variable(name='w3',shape=[3, 3, 192, 384])
                h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3, s=1))
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_conv3, [0], keep_dims=True)
                    h_conv3 = tf.nn.batch_normalization(
                        h_conv3, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv4') as scope:
                W_conv4 = self.weight_variable(name='w4',shape=[3,3,384,384])
                h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4,s=1))
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_conv4, [0], keep_dims=True)
                    h_conv4 = tf.nn.batch_normalization(
                        h_conv4, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('conv5') as scope:
                W_conv5 = self.weight_variable(name='w5',shape=[3,3,384,256])
                h_conv5 = tf.nn.relu(self.conv2d(h_conv4, W_conv5, s=1))
                h_pool5 = self.max_pool(h_conv5, k=3, s=2)
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_pool5, [0], keep_dims=True)
                    h_pool5 = tf.nn.batch_normalization(
                        h_pool5, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('fc6') as scope:
                fcw_init = tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
                W_fc6 = tf.get_variable(name='fc6_w',shape=[2, 2, 256, 4096], initializer=fcw_init)
                h_fc6 = tf.nn.relu(tf.nn.conv2d(h_pool5, W_fc6, strides=[1, 1, 1, 1], padding='VALID'))
                h_fc6_d = tf.nn.dropout(h_fc6, keep_prob=self.keep_prob)
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_fc6_d, [0], keep_dims=True)
                    h_fc6_d = tf.nn.batch_normalization(
                        h_fc6_d, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('fc7') as scope:
                W_fc7=tf.get_variable(name='fc7_w',shape=[1, 1, 4096, 4096],initializer=fcw_init)
                h_fc7 = tf.nn.relu(self.conv2d(h_fc6_d, W_fc7, s=1))
                h_fc7_d = tf.nn.dropout(h_fc7, keep_prob=self.keep_prob)
                if batch_norm:
                    batch_mean, batch_var = tf.nn.moments(
                        h_fc7_d, [0], keep_dims=True)
                    h_fc7_d = tf.nn.batch_normalization(
                        h_fc7_d, batch_mean, batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

            with tf.name_scope('fcout') as scope:
                fcout_init = tf.zeros_initializer()
                W_fc8 = tf.get_variable(name='fc8_w',shape=[1, 1, 4096, self.n_classes], initializer=fcout_init)
                h_fc8 = self.conv2d(h_fc7_d, W_fc8, s=1)
                self.output = tf.squeeze(h_fc8, [1, 2], name='fcout/squeezed')

