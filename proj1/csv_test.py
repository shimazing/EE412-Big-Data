import csv
import numpy as np
from classifiers.linear_classifier import LinearSVM
from sklearn import svm
from sklearn import tree




def binary_encoding(datum):
    #datum is length 10 1-dim array
    encoded = np.zeros(52)
    cards = datum.reshape(5, 2)
    for suit, rank in cards:
        encoded[(suit - 1)* 13 + rank - 1] = 1
    return encoded

def encoding_data(X):
    encoded_data = np.zeros((X.shape[0], 52))
    for i, datum in enumerate(X):
        encoded_data[i] = binary_encoding(datum)
    return encoded_data


with open ('training_modified.csv', 'r') as raw_file:
    reader = csv.reader(raw_file, delimiter=',')
    raw_data = np.array(list(reader)[1:]).astype(np.float32)

print(raw_data)
print(raw_data.shape)

# split data into test and train set
# test : train = 2 : 8 = 4000 : 16000
test_X = raw_data[:4000, :10]
train_X = raw_data[4000:, :10]

test_y = raw_data[:4000, 10].astype(np.int64)
train_y = raw_data[4000:, 10].astype(np.int64)

#for1 = test_X[test_y==1]
#for2 = test_X[test_y==2]

#print(for1[:, [0,2,4,6,8]])
#print(for2[:, [0,2,4,6,8]])


#test_X = encoding_data:q(test_X)
#train_X = encoding_data(train_X)


num_test, dim = test_X.shape
num_train = train_X.shape[0]

print(test_X)
print(train_X)
print(type(train_X[0][1]))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X, train_y)
dt_pred = clf.predict(test_X)

my_svm = LinearSVM()
history = my_svm.train(train_X, train_y, num_iters=800, verbose=True, reg=1e-6)
test_pred = my_svm.predict(test_X)

#kernel_svm = svm.SVC(kernel='poly', degree=2,
#        class_weight={0:1, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:10, 9:10},
#        verbose=True)

#kernel_svm.fit(train_X, train_y)
#svm_pred = kernel_svm.predict(test_X)


dt_accuracy = np.mean(test_y == dt_pred)
test_accuracy = np.mean(test_y == test_pred)
#svm_accuracy = np.mean(test_y == svm_pred)

print(test_accuracy)
print(dt_accuracy)
#print(svm_accuracy)


"""
with open('result.csv', 'w') as result_file:
    writer = csv.writer(result_file)
    writer.writerows(tmp)
"""
