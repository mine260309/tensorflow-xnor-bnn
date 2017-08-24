import tensorflow as tf
from tf_gemm_op import xnor_gemm
import numpy as ny


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)

# Alexnet with xnor for cifar10
class AlexBinaryNet:

    def __init__(self, binary, fast, n_hidden, keep_prob, x, batch_norm, phase):
        self.binary = binary
        self.fast = fast
        self.n_hidden = n_hidden
        self.keep_prob = keep_prob
        self.input = x
        self.G = tf.get_default_graph()
        self.conv_layers(batch_norm, phase)

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
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1,s,s,1], padding='SAME')

    def conv2d_1x1(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def conv_layers(self, batch_norm, phase):

        if self.binary:
            # based on the paper, all the bias numbers have been deleted.
            with tf.name_scope('conv1_bin') as scope:
                # don't quantize first layer
                W_conv1 = self.weight_variable('w1',shape=[11,11,1,64])
#                b1 = self.bias_variable('b1', shape=[64])
                h_conv1 = tf.nn.relu(self.conv2d(self.input, W_conv1, s=4))
                h_pool1 = self.max_pool(h_conv1, k=3, s=2)
                h_pool1_bin = self.binary_tanh_unit(h_pool1)

            with tf.name_scope('conv2_bin') as scope:

                W_conv2 = self.weight_variable(name='w2', shape=[5, 5, 64, 192])
#                b2 = self.bias_variable(name='b2', shape=[192])
                # compute the binary filters and scaling matrix
                Wb_conv2, alpha_2 = self.quantize_filter(W_conv2)
                # This is not a binary op right now...
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1_bin, Wb_conv2, s=1))
                # take max pool here to minimize quantization loss
                h_pool2 = self.max_pool(h_conv2, k=3, s=2)
                h_pool2_bin = self.binary_tanh_unit(h_pool2)

            with tf.name_scope('conv3_bin') as scope:

                W_conv3 = self.weight_variable(name='w3',shape=[3, 3, 192, 384])
#                b3 = self.bias_variable(name='b3', shape=[384])
                Wb_conv3, alpha_3 = self.quantize_filter(W_conv3)
                h_conv3 = tf.nn.relu(self.conv2d(h_pool2_bin, Wb_conv3, s=1))
                h_conv3_bin = self.binary_tanh_unit(h_conv3)

            with tf.name_scope('conv4_bin') as scope:

                W_conv4 = self.weight_variable(name='w4',shape=[3,3,384,384])
#                b4 = self.bias_variable(name='b4', shape=[384])
                Wb_conv4, alpha_4 = self.quantize_filter(W_conv4)
                h_conv4 = tf.nn.relu(self.conv2d(h_conv3_bin, Wb_conv4,s=1))
                h_conv4_bin = self.binary_tanh_unit(h_conv4)

            with tf.name_scope('conv5_bin') as scope:
                W_conv5 = self.weight_variable(name='w5',shape=[3,3,384,256])
                Wb_conv5, alpha_5 = self.quantize_filter(W_conv5)
#                b5 = self.bias_variable(name='b5', shape=[256])
                h_conv5 = tf.nn.relu(self.conv2d(h_conv4_bin, Wb_conv5))
                h_pool5 = self.max_pool(h_conv5, k=3, s=2)
                h_pool5_bin = self.binary_tanh_unit(tf.reshape(h_pool5, [-1, int(ny.prod(h_pool5.get_shape()[1:]))]))

            with tf.name_scope('fc6_bin') as scope:

                fcw_init = tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
#                fcb_init = tf.constant_initializer(0.1)

                W_fc6 = tf.get_variable(name='fc6_w',shape=[5,5,256,4096],initializer=fcw_init)
                Wb_fc6 = self.quantize(W_fc6)
#                b6 = tf.get_variable(name='fc6_b',shape=[4096],initializer=fcb_init)

                h_fc6 = tf.nn.relu(tf.matmul(h_pool5_bin, Wb_fc6))
                h_fc6_d = tf.nn.dropout(h_fc6, keep_prob=0.5) #how much portion we should keep?

            with tf.name_scope('fc7_bin') as scope:

                W_fc7=tf.get_variable(name='fc7_w',shape=[1,1,4096,4096],initializer=fcw_init)
                Wb_fc7 = self.quantize(W_fc7)
#                b7 = tf.get_variable(name='fc7_b', shape=[4096], initializer=fcb_init)
                h_fc7 = tf.nn.relu(tf.matmul(h_fc6_d, Wb_fc7))
                h_fc7_d = tf.nn.dropout(h_fc7, keep_prob=0.5)

            with tf.name_scope('fcout_bin') as scope:
                fcoutW = tf.get_variable(name='fc8_w',shape=[1,1,4096,20],initializer=fcw_init)
#                fc8b = tf.get_variable(name='fc8_b', shape=[20], initializer=fcb_init)
                fcout = tf.nn.relu(tf.matmul(h_fc7_d, fcoutW))
                self.output = tf.nn.softmax(fcout)

        else:
            ## TODO: normal alexnet
            with tf.name_scope('conv1_fp') as scope:

                W_conv1 = self.weight_variable([5, 5, 1, 32])
                h_conv1 = tf.nn.relu(self.conv2d(self.input, W_conv1))
                h_pool1 = self.max_pool_2x2(h_conv1)

            with tf.name_scope('conv2_fp') as scope:

                W_conv2 = self.weight_variable([5, 5, 32, 64])
                self.W_conv2_p = tf.reduce_sum(1.0 - tf.square(W_conv2))

                if batch_norm: # doesn't seem to help for full precision
                    h_pool1_batch_mean, h_pool1_batch_var = tf.nn.moments(
                        h_pool1, [0], keep_dims=True)
                    h_pool1 = tf.nn.batch_normalization(
                        h_pool1, h_pool1_batch_mean, h_pool1_batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)

                h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2))
                h_pool2 = self.max_pool_2x2(h_conv2)

            with tf.name_scope('fc1_fp') as scope:

                W_fc1 = self.weight_variable([7 * 7 * 64, self.n_hidden])
                h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
                '''
                if batch_norm:
                    h_pool2_flat_batch_mean, h_pool2_batch_var = tf.nn.moments(
                        h_pool2_flat, [0], keep_dims=True)
                    h_pool2 = tf.nn.batch_normalization(
                        h_pool2_flat, h_pool2_flat_batch_mean, h_pool2_batch_var, offset=None, scale=None, variance_epsilon=BN_EPSILON)
                '''
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1))

            with tf.name_scope('fcout_fp') as scope:

                W_fc2 = self.weight_variable([self.n_hidden, 10])
                h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
                self.output = tf.matmul(h_fc1_drop, W_fc2)


 #   def train(self, data, ):
