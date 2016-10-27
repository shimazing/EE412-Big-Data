import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import itertools

perm = itertools.permutations

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

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    f = open(filename,'r')
    for line in f:
        row = line.split(",")
        labels.append(int(row[10]))
        fvecs.append([float(x) for x in row[0:10]])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)

    # Convert the array of int labels into a numpy array.
    labels_np = np.array(labels).astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    # Return a pair of the feature matrix and the one-hot label matrix.
    #print(fvecs_np, labels_onehot)
    return fvecs_np,labels_onehot

def result_table(pred, ori):
    result = np.zeros((10,10)).astype(int)
    for i, j in zip(pred, ori):
        result[i,j] += 1

    with fullprint():
        print(result)

# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    elif init_method == 'positive':
        return tf.Variable(tf.random_normal(shape, mean=0.02, stddev=0.01,\
                                            dtype=tf.float32)) #0.02
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -0.1*np.sqrt(1.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high = 0.1*np.sqrt(1.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose

    # Get the data.
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test

    # Extract it into numpy arrays.
    train_data, train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename)

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
    w_hidden = init_weights(
        [num_features, num_hidden],
        'positive')
       # 'xavier',
       # xavier_params=(num_features, num_hidden))

    b_hidden = init_weights([1, num_hidden],'positive')

    # The hidden layer.
    hidden = tf.nn.relu(tf.matmul(x,w_hidden) + b_hidden)

    w_hidden2 = init_weights(
            [num_hidden, num_hidden2],
            'positive')
            # 'xavier',
            # xavier_params=(num_hidden, num_hidden2))

    b_hidden2 = init_weights([1,num_hidden2], 'positive')

    # The second hidden layer.
    hidden2 = tf.nn.relu(tf.matmul(hidden, w_hidden2) + b_hidden2)

    w_hidden3 = init_weights(
        [num_hidden2, num_hidden3],
        'positive')

    b_hidden3 = init_weights([1, num_hidden3], 'positive')

    # The third hidden layer
    hidden3 = tf.nn.relu(tf.matmul(hidden2, w_hidden3) + b_hidden3)

    # Initialize the output weights and biases.
    w_out = init_weights(
        [num_hidden2, NUM_LABELS],
        'positive')#,
        #xavier_params=(num_hidden2, NUM_LABELS))

    b_out = init_weights([1,NUM_LABELS],'positive')

    # The output layer.
    y = tf.nn.log_softmax(tf.matmul(hidden2, w_out) + b_out)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*y)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # Evaluation.
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
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
        print("Start training")

        # Iterate and train.
        for step in range(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print(step,)

            offset = (step * BATCH_SIZE) % train_size
            #print("index", offset, offset+BATCH_SIZE)
            choice = np.random.choice(range(train_size), BATCH_SIZE)
            #batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            #batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            batch_data = train_data[choice, :]
            batch_labels = train_labels[choice]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})
            if verbose and offset >= train_size-BATCH_SIZE:
                print(s.run([hidden, cross_entropy,y,tf.argmax(y_,1),accuracy],
                              feed_dict={x: batch_data, y_: batch_labels}))
            acc = accuracy.eval(feed_dict={x: batch_data, y_: batch_labels})
            if acc > 0.85 and step % 100 == 0:
                print(step, acc)
                pass

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


if __name__ == '__main__':
    tf.app.run()
