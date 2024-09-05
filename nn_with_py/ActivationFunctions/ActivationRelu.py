import numpy as np

class Activation_Relu:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        # Zero gradient
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs
