import numpy as np
from perceptron import perceptron


def Z_normalize(ARRAY):
    return (ARRAY-np.mean) / np.std(np.divide())

if __name__ == '__main__':
    # todo argv[0]
    X = np.genfromtxt("train_x.txt", delimiter=',')
    Y = np.genfromtxt("train_y.txt", delimiter=",")
    print(X, Y)
    X = Z_normalize(X)
    perceptron(X, Y)



