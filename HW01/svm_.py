import numpy as np
import matplotlib.pyplot as plt


class LinearSVM():

    def __init__(self):
        self.w = None

    def train(self, X, y, C, learning_rate, max_iter):

        '''
        train svm model using gradient descent


        Inputs:

        - X : shape (D, N)
        - y : shape (N,) 1 if class_A , -1 if class_B
        - C : loss is C*sum of all margins + 1/2||w||
        - learning_rate
        - max_iter

        '''
        dim, train_num = X.shape

        if self.w is None:
            self.w = np.array([1, 1, -1]) #np.random.randn(1, dim)

        for i in range(max_iter):
            loss, dw = self.svm_loss(X, y, C)
            self.w = self.w - learning_rate * dw


    def svm_loss(self, X, y, C):
        dim, train_num = X.shape
        scores = self.w.dot(X) # 1xN
        scores = scores * y
        margin = np.maximum(1 - scores, 0)
        loss = C * np.sum(margin)
        dmargin = np.zeros_like(margin)
        dmargin[margin > 0] = -1

        loss *= C
        loss += 0.5 * np.sum((self.w * self.w)[:dim - 1])
        tmp = X * y


        dw = dmargin.dot(tmp.T)
        dw = dw * C
        dw = dw + self.w
        dw[-1] -= self.w[-1]
        return loss, dw

if __name__ == '__main__':

    class_A = [(5, 7), (7, 3), (0, 13), (1, 8), (2, 5), (4, 2), (1, 3), (6, 0)]
    class_B = [(8, 6), (12, 3), (5, 13), (8, 9), (7, 16), (10, 7), (15, 5), (11, 11)]

    X = np.array([[x1, x2, 1] for x1 ,x2 in class_A + class_B]).T # 3 x 16
    y = np.hstack((-1 * np.ones(len(class_A)), np.ones(len(class_B))))

    print(X)
    print(y)

    cls = LinearSVM()
    cls.train(X, y, 100, 0.000001, 10000000)
    print(cls.w)

    print(cls.w.dot(X))

    a = - cls.w[0] / cls.w[1]
    b = - cls.w[2] / cls.w[1]

    print(a, b)
    A = np.array(class_A).T
    B = np.array(class_B).T
    t = np.arange(0., 20., 0.2)
    plt.plot(A[0], A[1], 'ro', B[0], B[1], 'bs', t, a*t + b)
    plt.axis([0,20,0,20])
    plt.show()
