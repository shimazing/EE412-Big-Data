import numpy as np
import csv

# global var
NUM_LABELS = 10

class fullprint:
    '''context manager for printing full numpy arrays'''

    def __enter__(self):
        self.opt = np.get_printoptions()
        np.set_printoptions(threshold=np.nan)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self.opt)

def binary_encoding(datum):

    encoded = np.zeros(52)
    cards = datum.reshape(5, 2)
    for suit, rank in np.array(cards):
        encoded[(suit - 1)* 13 + rank - 1] = 1
    return encoded

def encode_data(X):
    encoded_data = np.zeros((X.shape[0], 52))
    for i, datum in enumerate(X):
        encoded_data[i] = binary_encoding(datum)
    return encoded_data

def extract_data(filename, is_train=True):

    with open(filename, 'r') as raw_file:
        reader = csv.reader(raw_file, delimiter=',')
        raw_data = np.array(list(reader)[1:]).astype(np.float32)

    fvecs_np = raw_data[:, :10]
    if is_train:
        labels_np = raw_data[:, 10].astype(dtype=np.uint8)

    # Convert the int numpy array into a one-hot matrix.
    if is_train:
        labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)
    else:
        labels_onehot = None

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


def print_detail_result(pred, ori):
    result = np.zeros((10,10)).astype(int)
    for i, j in zip(pred, ori):
        result[i,j] += 1

    with fullprint():
        print(result)
