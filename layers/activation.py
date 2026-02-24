import numpy as np
from core.layer import Layer

class Activation(Layer):
    def __init__(self, activation_name):
        super().__init__()
        self.activation_name = activation_name
        
    def forward(self, input_data, training=True):
        self.input = input_data
        if self.activation_name == 'relu':
            self.output = np.maximum(0, self.input)
            
        elif self.activation_name == 'sigmoid':
            clipped = np.clip(self.input, -500, 500)
            self.output = 1 / (1 + np.exp(-clipped))
            
        elif self.activation_name == 'tanh':
            self.output = np.tanh(self.input)
            
        elif self.activation_name == 'softmax':
            shifted = self.input - np.max(self.input, axis=1, keepdims=True)
            exps = np.exp(shifted)
            self.output = exps / np.sum(exps, axis=1, keepdims=True)
            
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
            
        return self.output
        
    def backward(self, output_error):
        if self.activation_name == 'relu':
            derivative = (self.input > 0).astype(float)
            
        elif self.activation_name == 'sigmoid':
            sig = self.output
            derivative = sig * (1 - sig)
            
        elif self.activation_name == 'tanh':
            derivative = 1 - np.power(self.output, 2)
            
        elif self.activation_name == 'softmax':
            return output_error
            
        return output_error * derivative
