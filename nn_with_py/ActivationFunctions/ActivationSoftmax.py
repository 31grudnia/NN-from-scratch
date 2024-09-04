import numpy as np

class Activation_Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        
        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize for each sample
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probs

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradient
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calc jacobian matrix of the output
            jacobian_matrix  = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calc sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)