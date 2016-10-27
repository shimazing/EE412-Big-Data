import numpy as np
import tensorflow as tf

import tensorflow.python.platform
from helper import NUM_LABELS, extract_data, encode_data, fullprint


tf.app.flags.DEFINE_string('test', None,'Test file name')
tf.app.flags.DEFINE_string('ckpt', 'poker.ckpt', 'Trained model')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output')
tf.app.flags.DEFINE_integer('num_hidden', 40,
                                    'Number of nodes in the hidden layer.')

tf.app.flags.DEFINE_integer('num_hidden2', 25,
                                    'Number of nodes in the second hidden layer.')
FLAGS = tf.app.flags.FLAGS

def main(args=None):
    test_filename = FLAGS.test
    verbose = FLAGS.verbose

    # Get the size of layers
    num_hidden = FLAGS.num_hidden
    num_hidden2 = FLAGS.num_hidden2

    X, _ = extract_data(test_filename, is_train=False)
    print(X)
    X = encode_data(X)

    num_test, num_features = X.shape

    x = tf.placeholder('float', shape=[None, num_features])

    # 1st hidden layer
    w_hidden = tf.Variable(tf.zeros([num_features, num_hidden]), name='w_hidden')
    b_hidden = tf.Variable(tf.zeros([1, num_hidden]), name='b_hidden')
    hidden = tf.nn.relu(tf.matmul(x, w_hidden) + b_hidden)

    # 2nd hidden layer
    w_hidden2 = tf.Variable(tf.zeros([num_hidden, num_hidden2]), name='w_hidden2')
    b_hidden2 = tf.Variable(tf.zeros([1, num_hidden2]), name='b_hidden2')
    hidden2 = tf.nn.relu(tf.matmul(hidden, w_hidden2) + b_hidden2)

    # output layer
    w_out = tf.Variable(tf.zeros([num_hidden2, NUM_LABELS]), name='w_out')
    b_out = tf.Variable(tf.zeros([1, NUM_LABELS]), name='b_out')
    y = tf.nn.log_softmax(tf.matmul(hidden2, w_out) + b_out)

    # length 'num_test' list meaning hand of the cards set
    prediction = tf.argmax(y, 1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.ckpt)
        if verbose:
            print(">>> Trained model restored...")

        prediction = sess.run(prediction, feed_dict={x: X})
        with fullprint():
            print(prediction)


if __name__ == '__main__':
    tf.app.run()

