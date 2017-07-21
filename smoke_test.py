import time
import numpy as np
import tensorflow as tf

N = 4096
N_RUNS = 5

gemm_module = tf.load_op_library('./libs/gemm_op.so')

A = tf.placeholder(tf.float32, [N, N])
B = tf.placeholder(tf.float32, [N, N])
xnor_gemm = gemm_module.gemm(A, B)
matmul = tf.matmul(A,B)

# Re-use a for benchmarking on GPU w/only 4GB memory
a = 2 * tf.cast(tf.random_normal(shape=[N, N], seed=1) > 0, tf.float32) - 1

xnor_timings = np.zeros(N_RUNS)
base_timings = np.zeros(N_RUNS)

with tf.Session() as sess:

    a_f32 = sess.run(a)
    #b_f32 = sess.run(b)
    
    for i in range(N_RUNS):
        ########### benchmark xnor ############
        start_time = time.time()
        #xnor_gemm_result = sess.run(xnor_gemm, feed_dict={A: a_f32, B: a_f32})
        xnor_gemm_result = sess.run(gemm_module.gemm(a_f32, a_f32))
        xnor_timings[i] = time.time() - start_time

        print("xnor_gemm %d took %f" % (i, xnor_timings[i]))
        print(xnor_gemm_result)
        #######################################
    print("Avg XNOR kernel execution time over %d runs: %f +/- %f" % (N_RUNS, xnor_timings.mean(), xnor_timings.std()))

    for i in range(N_RUNS):
        ########### benchmark matmul ##########
        start_time = time.time()
        #matmul_result = sess.run(matmul, feed_dict={A: a_f32, B: a_f32})
        matmul_result = sess.run(tf.matmul(a_f32, a_f32))
        base_timings[i] = time.time() - start_time

        print("matmul %d took %f" % (i, base_timings[i]))
        print(matmul_result)
        #######################################
    print("Avg MatMul execution time over %d runs: %f +/- %f" % (N_RUNS, base_timings.mean(), base_timings.std()))
