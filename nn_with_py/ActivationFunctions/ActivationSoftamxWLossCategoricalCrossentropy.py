import numpy as np

from . import Loss_CategoricalCrossentropy
from .ActivationSoftmax import Activation_Softmax

# @TODO 
# Maybe add tests
class Activation_Softmax_Loss_CategoricalCross():
    def __init__(self) -> None:
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        sample_num = len(dvalues)

        # If One_hot, turn into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Calculate and normalize gradient
        self.dinputs = dvalues.copy()
        self.dinputs[range(sample_num), y_true] -= 1
        self.dinputs = self.dinputs / sample_num