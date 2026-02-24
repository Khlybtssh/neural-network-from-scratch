class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = False
        
    def forward(self, input_data, training=True):
        raise NotImplementedError
        
    def backward(self, output_error):
        raise NotImplementedError
