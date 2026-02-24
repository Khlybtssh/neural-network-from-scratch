import numpy as np

class LossFunction:
    def __init__(self, name):
        self.name = name
        
    def compute(self, y_true, y_pred):
        if self.name == 'mse':
            return np.mean(np.power(y_true - y_pred, 2))
            
        elif self.name == 'bce':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
        elif self.name == 'cce':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            
    def derivative(self, y_true, y_pred):
        if self.name == 'mse':
            return 2 * (y_pred - y_true) / y_true.size
            
        elif self.name == 'bce':
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / y_true.shape[0]
            
        elif self.name == 'cce':
            return (y_pred - y_true) / y_true.shape[0]
