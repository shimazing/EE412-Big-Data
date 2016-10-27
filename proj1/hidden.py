import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import csv

# Global variables.
NUM_LABELS = 10    # The number of labels.
BATCH_SIZE = 2000  # The number of training examples to use per training step.

tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', None,
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1500,
                            'Number of passes over the training data.')
tf.app.flags.DEFINE_integer('num_hidden', 35,
                            'Number of nodes in the hidden layer.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')

tf.app.flags.DEFINE_integer('num_hidden2', 30,
                            'Number of nodes in the second hidden layer.')
tf.app.flags.DEFINE_integer('num_hidden3', 15,
                            'Number of nodes in the third hidden layer.')
FLAGS = tf.app.flags.FLAGS

class fullprint:
    '''context manager for printing full numpy arrays'''

    def __enter__(self):
        self.opt = np.get_printoptions()
        np.set_printoptions(threshold=np.nan)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self.opt)

def binary_encoding(datum):
    #datum is length 10 1-dim array
    encoded = np.zeros(52)
    cards = datum.reshape(5, 2)
    for suit, rank in np.array(cards):
        encoded[(suit - 1)* 13 + rank - 1] = 1
    return encoded

def encoding_data(X):
    encoded_data = np.zeros((X.shape[0], 52))
    for i, datum in enumerate(X):
        encoded_data[i] = binary_encoding(datum)
    return encoded_data


#   label, feat_0, feat_1, ..., feat_n
def extract_data(filename):

    with open(filename, 'r') as raw_file:
        reader = csv.reader(raw_file, delimiter=',')
        raw_data = np.array(list(reader)[1:]).astype(np.float32)

    fvecs_np = raw_data[:, :10]
    labels_np = raw_data[:, 10].astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,labels_onehot

def split_train_test(X, y, p=0.1):
    train_size = X.shape[0]
    test_size = int(train_size * p)
    indices = np.random.permutation(train_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]

    test_X = X[test_idx,:]
    test_y = y[test_idx]

    train_X = X[train_idx,:]
    train_y = y[train_idx]

    return train_X, train_y, test_X, test_y


def result_table(pred, ori):
    result = np.zeros((10,10)).astype(int)
    for i, j in zip(pred, ori):
        result[i,j] += 1

    with fullprint():
        print(result)

# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='positive'):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    elif init_method == 'positive':
        return tf.Variable(tf.random_normal(shape, mean=0.02, stddev=0.01,\
                                            dtype=tf.float32)) #0.02

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose

    # Get the data.
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test

    # Extract it into numpy arrays.
    X, y = extract_data(train_data_filename)
    #test_data, test_labels = extract_data(test_data_filename)


    # randomly split train and test data with some rate p
    train_data, train_labels, test_data, test_labels = \
            split_train_test(X, y, p=0.2)

    # one hot encoding
    train_data = encoding_data(train_data)
    test_data = encoding_data(test_data)

    # Get the shape of the training data.
    train_size,num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the size of layer one.
    num_hidden = FLAGS.num_hidden
    num_hidden2 = FLAGS.num_hidden2
    num_hidden3 = FLAGS.num_hidden3

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])

    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # Define and initialize the network.

    # Initialize the hidden weights and biases.

    # Make 1st hidden layer and op
    w_hidden = init_weights([num_features, num_hidden], 'positive')
    b_hidden = init_weights([1, num_hidden],'positive')
    hidden = tf.nn.relu(tf.matmul(x,w_hidden) + b_hidden)

    # Make 2nd hidden layer and op
    w_hidden2 = init_weights([num_hidden, num_hidden2],'positive')
    b_hidden2 = init_weights([1,num_hidden2], 'positive')
    hidden2 = tf.nn.relu(tf.matmul(hidden, w_hidden2) + b_hidden2)

    # Make 3rd hidden layer and op
    w_hidden3 = init_weights([num_hidden2, num_hidden3],'positive')
    b_hidden3 = init_weights([1, num_hidden3], 'positive')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w_hidden3) + b_hidden3)

    # Output layer
    w_out = init_weights([num_hidden2, NUM_LABELS],'positive')
    b_out = init_weights([1,NUM_LABELS],'positive')
    y = tf.nn.log_softmax(tf.matmul(hidden2, w_out) + b_out)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*y)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init_op = tf.initialize_all_variables()

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        init_op.run()

        if verbose:
            print('Initialized!')
            print()
            print('Training.')

        with fullprint():
            print(s.run(w_hidden))
            print(s.run(b_hidden))
            print(s.run(w_hidden2))
            print(s.run(b_hidden2))
            print(s.run(w_out))
            print(s.run(b_out))

        if verbose:
            print("Start training")

        # Iterate and train.
        for step in range(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print(step,)

            choice = np.random.choice(range(train_size), BATCH_SIZE)
            batch_data = train_data[choice, :]
            batch_labels = train_labels[choice]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})

            if verbose:
                acc = accuracy.eval(feed_dict={x: batch_data, y_: batch_labels})
                if acc > 0.85 and step % 100 == 0:
                    print(step, acc)

        # Print out train and test accuracy
        print("Train Accuracy:", accuracy.eval(feed_dict={x: train_data, y_:\
            train_labels}))
        print("Accuracy:", accuracy.eval(feed_dict={x: test_data, y_:\
            test_labels}))

        # result table
        pred_train_y, real_train_y = s.run([tf.argmax(y,1),tf.argmax(y_,1)],\
                feed_dict={x: train_data, y_: train_labels})
        pred_test_y, real_test_y = s.run([tf.argmax(y,1), tf.argmax(y_,1)],\
                feed_dict={x: test_data, y_: test_labels})

        result_table(pred_train_y, real_train_y)
        result_table(pred_test_y, real_test_y)

        # save trained model

        saver = tf.train.Saver()
        save_path = saver.save(s, 'poker.ckpt')
        print('poker hand classifier saved in file {}'.format(save_path))

if __name__ == '__main__':
    tf.app.run()
