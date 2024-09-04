import numpy as np

from .Loss import Loss

# @TODO tests
# Classification problems with 2 outputs
class Loss_BinaryCrossentropy(Loss):
    
    def froward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) \
                          * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        sample_num = len(dvalues)
        output_num = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - \
                         (1 - y_true) / (1 - clipped_dvalues)) / output_num
        # Normalize gradient
        self.dinputs = self.dinputs / sample_num 