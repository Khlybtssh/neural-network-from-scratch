import numpy as np

class Optimizer:
    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    """Vanilla Gradient Descent / SGD with optional Momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}
        
    def update(self, layer):
        if not hasattr(layer, 'trainable') or not layer.trainable:
            return
            
        if layer not in self.velocities:
            self.velocities[layer] = {'W': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.bias)}
            
        v_W = self.momentum * self.velocities[layer]['W'] - self.lr * layer.dW
        v_b = self.momentum * self.velocities[layer]['b'] - self.lr * layer.db
        
        self.velocities[layer]['W'] = v_W
        self.velocities[layer]['b'] = v_b
        
        layer.weights += v_W
        layer.bias += v_b
