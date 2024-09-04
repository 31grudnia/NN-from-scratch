import numpy as np

from .Loss import Loss

# Regression problems
class Loss_MeanSquaredError(Loss): #L2 Loss
    
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        sample_num = len(dvalues)
        output_num = len(dvalues[0])

        # Gradient on vals 
        self.dintputs = -2 * (y_true - dvalues) / output_num
        # Normalize grad
        self.dintputs = self.dintputs / sample_num