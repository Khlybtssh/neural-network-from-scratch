import numpy as np
from core.layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, initialization='he', l2_lambda=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.l2_lambda = l2_lambda
        
        if initialization == 'random_normal':
            self.weights = np.random.randn(input_size, output_size) * 0.01
        elif initialization == 'xavier':
            stddev = np.sqrt(2.0 / (input_size + output_size))
            self.weights = np.random.randn(input_size, output_size) * stddev
        elif initialization == 'he':
            stddev = np.sqrt(2.0 / input_size)
            self.weights = np.random.randn(input_size, output_size) * stddev
        else:
            raise ValueError(f"Unknown initialization: {initialization}")
            
        self.bias = np.zeros((1, output_size))
        
        self.dW = np.zeros_like(self.weights)
        self.db = np.zeros_like(self.bias)
        
    def forward(self, input_data, training=True):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
        
    def backward(self, output_error):
        m = self.input.shape[0]
        
        self.dW = np.dot(self.input.T, output_error) / m
        
        if self.l2_lambda > 0:
            self.dW += (self.l2_lambda / m) * self.weights
            
        self.db = np.sum(output_error, axis=0, keepdims=True) / m
        
        input_error = np.dot(output_error, self.weights.T)
        return input_error
