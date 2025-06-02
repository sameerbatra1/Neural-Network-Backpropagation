import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(y_true, y_pred):
    loss = 0.5 * (y_true - y_pred) ** 2
    return round(loss, 3)
