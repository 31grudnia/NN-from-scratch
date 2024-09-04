import numpy as np

from .Accuracy import Accuracy


class Accuracy_Regression(Accuracy):

    def __init__(self) -> None:
        self.precision = None

    
    def init(self, Y, reinit=False):
        
        if self.precision is None or reinit:
            self.precision = np.std(Y) / 250
        

    def comapre(self, predictions, Y):
        return  np.absolute(predictions - Y) < self.precision