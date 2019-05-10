import numpy as np
from random import shuffle


def perceptron(X_train, Y_train):
    m = len(X_train)
    d = len(X_train[0])
    eta = 0.1
    w = np.zeros((d,))

    # perceptron
    # multi class
    epochs = 20
    for e in range(epochs):
        X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
        for x, y in zip(X_train, Y_train):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            # update
            if y != y_hat:
                w[y, :] += eta * x
                w[y_hat, :] -= eta * x
    w_perceptron = w

    # testing
    m_perceptron = 0
    for t in range(0, m):
        y_hat = np.sign(np.dot(w_perceptron, X_train[t]))
        if Y_train[t] != y_hat:
            m_perceptron += 1
    print("perceptron err =", float(m_perceptron)/m)


