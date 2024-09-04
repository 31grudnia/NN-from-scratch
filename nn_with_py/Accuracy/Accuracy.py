import numpy as np


class Accuracy:

    def calculate(self, predictions, Y):
        
        comparisions = self.compare(predictions, Y)
        accuracy = np.mean(comparisions)

        self.accumulated_sum += np.sum(comparisions)
        self.accumulated_count += len(comparisions)

        return accuracy
    

    def calculate_accumulated(self):

        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
