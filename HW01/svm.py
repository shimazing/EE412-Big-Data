import numpy as np
import matplotlib as m
class_A = [(5,7), (7,3), (0,13), (1,8), (2,5), (4,2), (1,3), (6,0)]
class_B = [(8, 6), (12, 3), (5, 13), (8, 9), (7, 16), (10, 7), (15, 5), (11, 11)]

data = [(x1, x2, -1) for x1, x2 in class_A] + [(x1, x2, 1) for x1, x2 in class_B]
n = len(data)

# Init w = (w1, w2) , b
w1 = 1
w2 = 1
b = -1

# learning rate
alpha = 0.001
# regularization parameter
C = 0.5
# when converge?
thrs = 0.0001

def value(x1, x2, y):
    global w1, w2, b
    return (w1*x1 + w2*x2 + b)*y

def loss(x1, x2, y):
    global w1, w2, b
    return max(1 - (w1*x1 + w2*x2 + b)*y, 0)

def round_w1(x1, x2, y):
    if loss(x1, x2, y):
        return - x1 * y
    return 0

def round_w2(x1, x2, y):
    if loss(x1, x2, y):
        return - x2 * y
    return 0

def round_b(x1, x2, y):
    if loss(x1, x2, y):
        return - y
    return 0

def new_w1(data, alpha):
    return w1 - alpha * (C * np.sum([round_w1(x1, x2, y) for x1, x2, y in data]) \
                        + w1)
def new_w2(data, alpha):
    return w2 - alpha * (C * np.sum([round_w2(x1, x2, y) for x1, x2, y in data]) \
                        + w2)
def new_b(data, alpha):
    return b - alpha * (C * np.sum([round_b(x1, x2, y) for x1, x2, y in data]))


def gradient_descent(data):
    global w1, w2, b
    converge = False
    iter = 1
    while not converge:
        w1_n = new_w1(data, 10/iter)
        w2_n = new_w2(data, 10/iter)
        b_n = new_b(data, 10/iter)
        if abs(w1_n - w1) < thrs and abs(w2_n - w2) < thrs and abs(b_n - b) <thrs:
            converge = True
            print("converge")
        w1 = w1_n
        w2 = w2_n
        b = b_n
        iter += 1
        if iter > 100000:
            break

gradient_descent(data)
print(w1, w2, b)
for x1, x2, y in data:
    print(loss(x1, x2, y), value(x1, x2, y))
