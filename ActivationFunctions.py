import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivation_sigmoid(a):
    return a * (1 - a)

def ReLU(x):
    return np.maximum(0, x)