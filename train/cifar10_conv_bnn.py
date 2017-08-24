
import argparse
import tensorflow as tf
from models.alexnet import AlexBinaryNet as AlexNet
from importData import Dataset

BN_TRAIN_PHASE = True
BN_TEST_PHASE = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='directory for storing input data')
    parser.add_argument(
        '--log_dir', help='root path for logging events and checkpointing')
    parser.add_argument(
        '--keep_prob', help='dropout keep_prob', type=float, default=0.8)
    parser.add_argument(
        '--reg', help='how much to push weights to +1/-1', type=float, default=0.5)
    parser.add_argument(
        '--lr', help='learning rate', type=float, default=1e-5)
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


    # import data
    trainingData = Dataset(imagePath='./data/train/', extensions='jpg')
    testData     = Dataset(imagePath='./data/test/', extensions='jpg')

    lr = args.lr
    decay_rate = 0.1
    batch_size = args.batch_size
    display_step = 50
    dtype = tf.float32

    n_classes = trainingData.num_labels
    imagesize = 227
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
        pred = AlexNet(x, keep_prob, n_classes=n_classes, imagesize=imagesize, img_channel=img_channel)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(lr, global_step, 1000, decay_rate, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate= lr).minimize(cost, global_step=global_step)

        correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype))

        init = tf.initialize_all_variables()
        saver = tf.train.Saver();
        tf.add_to_collection("x", x)
        tf.add_to_collection("y", y)
        tf.add_to_collection("keep_prob", keep_prob)
        tf.add_to_collection("pred", pred)
        tf.add_to_collection("accuracy", accuracy)

        with tf.Session() as sess:
            sess.run(init)
            step = 1
            while step < maxsteps:
                batch_ys, batch_xs = trainingData.nextBatch(batch_size)
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    rate = sess.run(lr)
                    print 'learning rate ' + str(rate) + \
                                   ' Iter '+ str(step) + ' loss= '+ \
                                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

                if step % 1000 == 0:
                    saver.save(sess, 'save/model.ckpt', global_step=step*batch_size)
                    step += 1

            print " training is done "

            step_test = 1
            while step_test * batch_size < len(testData):
                testing_ys, testing_xs = testData.nextBatch(batch_size)
                print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: testing_xs, y: testing_ys, keep_prob: 1.})
        step_test += 1




