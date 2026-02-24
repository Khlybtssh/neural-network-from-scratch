import numpy as np
from optimizers.sgd import Optimizer

class Adam(Optimizer):
    """Adaptive Moment Estimation Optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        
    def increment_t(self):
        self.t += 1
        
    def update(self, layer):
        if not hasattr(layer, 'trainable') or not layer.trainable:
            return
            
        if layer not in self.m:
            self.m[layer] = {'W': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.bias)}
            self.v[layer] = {'W': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.bias)}
            
        self.m[layer]['W'] = self.beta1 * self.m[layer]['W'] + (1 - self.beta1) * layer.dW
        self.m[layer]['b'] = self.beta1 * self.m[layer]['b'] + (1 - self.beta1) * layer.db
        
        self.v[layer]['W'] = self.beta2 * self.v[layer]['W'] + (1 - self.beta2) * (layer.dW ** 2)
        self.v[layer]['b'] = self.beta2 * self.v[layer]['b'] + (1 - self.beta2) * (layer.db ** 2)
        
        m_hat_W = self.m[layer]['W'] / (1 - self.beta1 ** self.t)
        m_hat_b = self.m[layer]['b'] / (1 - self.beta1 ** self.t)
        
        v_hat_W = self.v[layer]['W'] / (1 - self.beta2 ** self.t)
        v_hat_b = self.v[layer]['b'] / (1 - self.beta2 ** self.t)
        
        layer.weights -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
