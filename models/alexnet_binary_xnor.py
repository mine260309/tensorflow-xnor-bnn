import tensorflow as tf
from tf_gemm_op import xnor_gemm

BN_EPSILON = 1e-5
H = 0
W = 1
INPUTS = 2
FILTERS = 3


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

    '''
    def init_layer(self, name, n_inputs, n_outputs):

        W = tf.get_variable(name, shape=[
                             n_inputs, n_outputs], initializer=tf.contrib.layers.xavier_initializer())
        # b = tf.Variable(tf.zeros([n_outputs]))
        return W
    '''

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

    '''
    def binarize_conv_input(conv_input, k):

        # This is from BinaryNet.
        # This acts like sign function during forward pass. and like hard_tanh
        # during back propagation
        bin_conv_out = self.binary_tanh_unit(conv_input)

        # scaling factor for the activation.
        A = tf.abs(conv_input)

        # K will have scaling matrixces for each input in the batch.
        # K's shape = (batch_size, 1, map_height, map_width)
        # K's tensorflow shape = (batchsize, h, w, 1
        #k_shape = k.eval().shape
        #k_shape = tf.shape(k)
        #pad = (k_shape[-3] // 2, k_shape[-2] // 2)
        # support the kernel stride. This is necessary for AlexNet
        K = theano.tensor.nnet.conv2d(A, k, border_mode=pad)

        return bin_conv_out, K
    '''

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

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

            with tf.name_scope('conv1_bin') as scope:
                # don't quantize first layer
                W_conv1 = self.weight_variable([3, 3, 1, 64])
                #self.W_conv1_p = tf.reduce_sum(1.0 - tf.square(W_conv1))
                self.W_conv1_summ = tf.summary.histogram(name='W_conv1_summ', values=W_conv1)

                h_conv1 = tf.nn.relu(self.conv2d(self.input, W_conv1, s=4))
                h_pool1 = self.max_pool(h_conv1, k=3, s=2)
#                self.h_pool1_summ = tf.summary.histogram(
#                    name='h_pool1_summ', values=h_pool1)

            with tf.name_scope('conv2_bin') as scope:

                W_conv2 = self.weight_variable([3, 3, 64, 128])
#                self.W_conv2_summ = tf.summary.histogram(name='W_conv2_summ', values=W_conv2)
#                self.W_conv2_p = tf.reduce_sum(1.0 - tf.square(W_conv2))
#
#                shape = tf.shape(h_pool1)
#
#                if batch_norm:
#                    h_pool1 = tf.contrib.layers.batch_norm(
#                        h_pool1, decay=0.9, center=False, scale=False, epsilon=BN_EPSILON, is_training=phase)
#                # compute the binary inputs H and the scaling matrix K
                h_pool1_bin = self.binary_tanh_unit(h_pool1)

                # compute the binary filters and scaling matrix
                Wb_conv2, alpha_2 = self.quantize_filter(W_conv2)
#                self.Wb_conv2_summ = tf.summary.histogram(
#                    name='Wb_conv2_summ', values=Wb_conv2)

                # This is not a binary op right now...
                h_conv2 = tf.nn.relu(self.conv2d(h_pool1_bin, Wb_conv2, s=1))

                # take max pool here to minimize quantization loss
                h_pool2 = self.max_pool(h_conv2, k=3, s=2)
#                self.h_pool2_summ = tf.summary.histogram(
#                    name='h_pool2_summ', values=h_pool2)

            with tf.name_scope('conv3_bin') as scope:

                h_pool2_bin = self.binary_tanh_unit(h_pool2)
                W_conv3 = self.weight_variable([3, 3, 128, 256])
                Wb_conv3, alpha_3 = self.quantize_filter(W_conv3)
                h_conv3 = tf.nn.relu(self.conv2d(h_pool2_bin, Wb_conv3, s=1))

            with tf.name_scope('conv4_bin') as scope:

                h_conv3_bin = self.binary_tanh_unit(h_conv3)
                W_conv4 = self.weight_variable([4*4*256, 1024])
                Wb_conv4, alpha_4 = self.quantize_filter(W_conv4)
                h_conv4 = tf.nn.relu(self.conv2d(h_conv3, Wb_conv4,s=1))

            with tf.name_scope('conv5_bin') as scope:
                W_conv5 = self.weight_variable([1024,1024])
                Wb_conv5, alpha_5 = self.quantize_filter(W_conv5)
                h_conv5 = tf.nn.relu(self.conv2d(h_conv4, Wb_conv5))
                h_pool5 = self.max_pool(h_conv5, k=3, s=2)

            with tf.name_scope('fc6_bin') as scope:

                W_fc6 = self.weight_variable([4 * 4 * 256, 1024])
                Wb_fc6 = self.quantize(W_fc6)

                h_pool_5 = tf.reshape(h_pool5, [-1, 4 * 4 * 256])
                h_fc6 = tf.nn.relu(tf.matmul(h_pool_5, Wb_fc6))

                h_fc6_d = tf.nn.dropout(h_fc6, self.keep_prob)

            with tf.name_scope('fc7_bin') as scope:

                W_fc7 = self.weight_variable([1024, 1024])
                Wb_fc7 = self.quantize(W_fc7)
                h_fc7 = tf.nn.relu(tf.matmul(h_fc6_d, Wb_fc7))
                h_fc7_d = tf.nn.dropout(h_fc7, self.keep_prob)


            with tf.name_scope('fcout_bin') as scope:

                W_fcout = self.weight_variable([4096, 10])
                Wb_fcout = self.quantize_filter(W_fcout)

                self.output = tf.nn.relu(tf.matmul(h_fc7_d, Wb_fcout))

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
