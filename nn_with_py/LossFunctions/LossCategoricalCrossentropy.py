import numpy as np

from .Loss import Loss


class Loss_CategoricalCrossentropy(Loss):

    def forwad(self, y_pred, y_true):
        samples_num = len(y_pred)
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Porbabilities:
        # For categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples_num), y_true]
        # For OneHot Encoding labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Losses
        cat_cross_loss = -np.log(correct_confidences)
        return cat_cross_loss
    
    def backward(self, dvalues, y_true):
        samples_num = len(dvalues)
        labels_num = len(dvalues[0])

        # If labels are sparse, turn them into OneHot Encoding vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels_num)[y_true] # cool ass line

        # Gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples_num
        