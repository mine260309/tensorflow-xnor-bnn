from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
from models.alexnet_binary_xnor import AlexBinaryNet as AlexNet
from importData import Dataset
import numpy as np

BN_TRAIN_PHASE = True
BN_TEST_PHASE = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory for storing input data')
    parser.add_argument(
        '--log_dir', help='root path for logging events and checkpointing')
    parser.add_argument(
        '--extra', help='for specifying extra details (e.g one-off experiments)')
    parser.add_argument(
        '--n_hidden', help='number of hidden units', type=int, default=512)
    parser.add_argument(
        '--keep_prob', help='dropout keep_prob', type=float, default=0.8)
    parser.add_argument(
        '--reg', help='how much to push weights to +1/-1', type=float, default=0.5)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument(
        '--batch_size', help='examples per mini-batch', type=int, default=128)
    parser.add_argument(
        '--max_steps', help='maximum training steps', type=int, default=1000)
    parser.add_argument(
        '--gpu', help='physical id of GPUs to use')
    parser.add_argument(
        '--eval_every_n', help='validate model every n steps', type=int, default=100)
    parser.add_argument(
        '--binary', help="should weights and activations be constrained to -1, +1", action="store_true")
    parser.add_argument(
        '--xnor', help="if binary flag is passed, determines if xnor_gemm cuda kernel is used to accelerate training, otherwise no effect", action="store_true")
    parser.add_argument(
        '--batch_norm', help="batch normalize activations", action="store_true")
    parser.add_argument(
        '--debug', help="run with tfdbg", action="store_true")
    parser.add_argument(
        '--restore', help='where to load model checkpoints from')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # handle command line args
    if args.binary:
        print("Using 1-bit weights and activations")
        binary = True
        sub_1 = '/bin/'
        if args.xnor:
            print("Using xnor xnor_gemm kernel")
            xnor = True
            sub_2 = 'xnor/'
        else:
            sub_2 = 'matmul/'
            xnor = False
    else:
        sub_1 = '/fp/'
        sub_2 = ''
        binary = False
        xnor = False

    if args.log_dir:
        log_path = args.log_dir + sub_1 + sub_2 + \
            'hid_' + str(args.n_hidden) + '/'

    if args.batch_norm:
        print("Using batch normalization")
        batch_norm = True
        alpha = 0.1
        epsilon = 1e-4
        if args.log_dir:
            log_path += 'batch_norm/'
    else:
        batch_norm = False

    if args.log_dir:
        log_path += 'bs_' + str(args.batch_size) + '/keep_' + \
            str(args.keep_prob) + '/reg_' + \
            str(args.reg) + '/lr_' + str(args.lr) + '/' + \
            args.extra
        log_path = create_dir_if_not_exists(log_path)

    # import data
    trainingData = Dataset(imagePath=args.data_dir + '/train/', extensions='.png')
    testData     = Dataset(imagePath=args.data_dir + '/test/', extensions='.png')

    lr = args.lr
    decay_rate = 0.1
    batch_size = args.batch_size
    display_step = 1
    dtype = tf.float32

    n_classes = trainingData.num_labels
    imagesize = 32
    img_channel = 3
    maxsteps = args.max_steps
    dropout = 0.8

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        x = tf.placeholder(dtype, [None, imagesize, imagesize, img_channel])
        y = tf.placeholder(dtype, [None, n_classes])
        phase = tf.placeholder(dtype, [None, n_classes])
        keep_prob = tf.placeholder(tf.float32)

        # create the model
        pred = AlexNet(binary, x, keep_prob, n_classes, imagesize, img_channel, phase)
        output = pred.output
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))

        global_step = tf.Variable(0, trainable=False)
        #lr = tf.train.exponential_decay(lr, global_step, 1000, decay_rate, staircase=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate= lr).minimize(cost, global_step=global_step)

        train_op = tf.contrib.layers.optimize_loss(
            cost, global_step, learning_rate=args.lr, optimizer='Adam',
            summaries=["gradients"])

        correct_pred = tf.equal(tf.arg_max(output, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype))

        init = tf.initialize_all_variables()
        saver = tf.train.Saver();
        tf.add_to_collection("x", x)
        tf.add_to_collection("y", y)
        tf.add_to_collection("keep_prob", keep_prob)
        tf.add_to_collection("pred", output)
        tf.add_to_collection("accuracy", accuracy)

        with tf.Session() as sess:
            sess.run(init)
            step = 1
            while step < maxsteps:
                batch_ys, batch_xs = trainingData.nextBatch(batch_size)
                __, loss = sess.run([train_op, cost], feed_dict={
                    x: batch_xs, y: batch_ys, keep_prob: dropout})

                #sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    print('learning rate ' + str(lr) + \
                                   ' Iter '+ str(step) + ' loss= '+ \
                                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

                if step % 1000 == 0:
                    saver.save(sess, 'model.ckpt', global_step=step*batch_size)
                step = step + 1

            print("training is done")

            step_test = 1
            while step_test * batch_size < len(testData):
                testing_ys, testing_xs = testData.nextBatch(batch_size)
                print("Testing Accuracy: %.4f" % (sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})))
                step_test += 1

'''
    #Inference Parameters
    validation = Dataset('/home/xnor/data/cifar10img/val/', '.png')
    batch_size = 1

    ckpt = tf.train.get_checkpoint_state("save")
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

    pred = tf.get_collection("pred")[0]
    x = tf.get_collection("x")[0]
    keep_prob = tf.get_collection("keep_prob")[0]

    # Launch the graph
    # with tf.Session() as sess:
    sess = tf.Session()
    saver.restore(sess, ckpt.model_checkpoint_path)

        # inferences
    step_test = 1
    while step_test * batch_size < len(validation):
        testing_ys, testing_xs = validation.nextBatch(batch_size)
        predict = sess.run(pred, feed_dict={x: testing_xs, keep_prob: 1.})
        print("Testing label:")
        print(validation.label2category[np.argmax(testing_ys, 1)[0]])
        print("Testing predict:")
        print(validation.label2category[np.argmax(predict, 1)[0]])
    step_test += 1
'''