import numpy as np

from .Loss import Loss

# Regression problems
class Loss_MeanAbsoluteError(Loss): #L1 Loss
    
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses
    
    def backward(self, dvalues, y_true):
        sample_num = len(dvalues)
        output_num = len(dvalues[0])

        # Gradient on vals 
        self.dintputs = np.sign(y_true - dvalues) / output_num
        # Normalize grad
        self.dintputs = self.dintputs / sample_num
