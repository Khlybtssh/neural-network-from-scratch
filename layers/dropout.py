import numpy as np
from core.layer import Layer

class Dropout(Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None
        
    def forward(self, input_data, training=True):
        self.input = input_data
        if training and self.rate > 0:
            self.mask = (np.random.rand(*input_data.shape) > self.rate) / (1.0 - self.rate)
            self.output = input_data * self.mask
        else:
            self.output = input_data
        return self.output
        
    def backward(self, output_error):
        if self.rate > 0:
            return output_error * self.mask
        return output_error
