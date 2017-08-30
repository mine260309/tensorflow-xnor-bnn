from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf
from models.alexnet_binary_xnor import AlexBinaryNet as AlexNet
from importData import Dataset
from utils import create_dir_if_not_exists
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
    parser.add_argument(
        '--validation', help='Run validation instead of training', action="store_true")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # handle command line args
    if args.binary:
        print("Using 1-bit weights and activations")
        binary = True
        sub_1 = 'bin'
        if args.xnor:
            print("Using xnor xnor_gemm kernel")
            xnor = True
            sub_2 = 'xnor'
        else:
            sub_2 = 'matmul'
            xnor = False
    else:
        sub_1 = 'fp'
        sub_2 = ''
        binary = False
        xnor = False

    if args.log_dir:
        log_path = os.path.join(args.log_dir, sub_1, sub_2)
        log_path = create_dir_if_not_exists(log_path)
    else:
        log_path = "log"

    checkpoint_path = os.path.join(log_path, 'model.ckpt')

    if args.batch_norm:
        print("Using batch normalization")
        batch_norm = True
        alpha = 0.1
        epsilon = 1e-4
        if args.log_dir:
            log_path += 'batch_norm/'
    else:
        batch_norm = False

    # Check parameters
    if args.validation and args.restore == None:
        print('--restore is required for validation')
        exit(1)

    # import data
    dataset_dir = args.data_dir
    trainingData = Dataset(os.path.join(dataset_dir, 'train'), '.png')
    testData = Dataset(os.path.join(dataset_dir, 'test'), '.png')
    validationData = Dataset(os.path.join(dataset_dir, 'val'), '.png')

    lr = args.lr
    batch_size = args.batch_size
    display_step = args.eval_every_n
    dtype = tf.float32

    n_classes = trainingData.num_labels
    imagesize = 32
    img_channel = 3
    maxsteps = args.max_steps
    dropout = args.keep_prob

    if args.validation:
        ckpt = tf.train.get_checkpoint_state(args.restore)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

        pred = tf.get_collection("pred")[0]
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        keep_prob = tf.get_collection("keep_prob")[0]
        accuracy = tf.get_collection("accuracy")[0]

        # Launch the graph
        # with tf.Session() as sess:
        sess = tf.Session()
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Validation
        step_test = 1
        while step_test < len(validationData):
            testing_ys, testing_xs = validationData.nextBatch(batch_size)
            acc = sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})
            print('Validation {:d} to {:d}, Accuracy= {:.6f}'.format(step_test, step_test + batch_size, acc))
            step_test = step_test + batch_size

        # inferences
        #step_test = 1
        #while step_test * batch_size < len(validationData):
        #    testing_ys, testing_xs = validationData.nextBatch(batch_size)
        #    predict = sess.run(pred, feed_dict={x: testing_xs, keep_prob: 1.})
        #    print("Testing label:")
        #    print(validationData.label2category[np.argmax(testing_ys, 1)[0]])
        #    print("Testing predict:")
        #    print(validationData.label2category[np.argmax(predict, 1)[0]])
        #step_test += 1
        exit(0)

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
            while step <= maxsteps:
                batch_ys, batch_xs = trainingData.nextBatch(batch_size)
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    val_ys, val_xs = validationData.nextBatch(batch_size)
                    val_acc = sess.run(accuracy, feed_dict={x: val_xs, y: val_ys, keep_prob: 1.})
                    print('learning rate {:.6f} Iter {:d} loss= {:.6f}, Training Accuracy= {:.6f}, Validation Accuracy= {:.6f}'
                          .format(lr, step, loss, acc, val_acc))

                if step % 1000 == 0:
                    saver.save(sess, checkpoint_path, global_step=step*batch_size)
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
