import numpy as np

def compute_accuracy(y_true, y_pred, is_categorical=True):
    if is_categorical:
        return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    else:
        return np.mean((y_pred > 0.5) == y_true)
