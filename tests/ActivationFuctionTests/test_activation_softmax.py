import numpy as np
import unittest

from . import Activation_Softmax


class TestActivationSoftmax(unittest.TestCase):

    def test_forward(self):
        softmax = Activation_Softmax()

        inputs = np.array([[1, 2, 3], [1, 2, 3]])
        softmax.forward(inputs)
        
        expected_output = np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096]
        ])
        
        np.testing.assert_allclose(softmax.output, expected_output, rtol=1e-5, err_msg="Softamx Forward pass failed.")

    def test_backward(self):
        softmax = Activation_Softmax()

        inputs = np.array([[1, 2, 3], [1, 2, 3]])
        softmax.forward(inputs)

        dvalues = np.array([[1, 2, 3], [1, 2, 3]])
        softmax.backward(dvalues)
        
        self.assertEqual(softmax.dinputs.shape, inputs.shape, "Softamx Backward pass shape mismatch.")
        

    def test_predictions(self):
        softmax = Activation_Softmax()

        outputs = np.array([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        predictions = softmax.predictions(outputs)
        
        expected_predictions = np.array([2, 1])
        
        np.testing.assert_array_equal(predictions, expected_predictions, err_msg="Softamx Predictions failed.")
