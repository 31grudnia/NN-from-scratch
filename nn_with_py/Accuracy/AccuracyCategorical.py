import numpy as np

from .Accuracy import Accuracy


class Accuracy_Categorical(Accuracy):

    def init(self, Y) -> None:
        pass


    def compare(self, predictions, Y):
        
        if len(Y.shape) == 2:
            Y = np.argmax(Y, axis=1)
        
        return predictions == Y