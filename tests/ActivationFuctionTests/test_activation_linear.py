import numpy as np
import unittest

from . import Activation_Linear


class TestActivationLinear(unittest.TestCase):

    def test_forward(self):
        activation = Activation_Linear()

        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        activation.forward(inputs)

        # The output of a linear activation function is the same as the input
        expected_output = inputs
        
        np.testing.assert_array_equal(activation.output, expected_output, err_msg="Forward pass failed.")

    def test_backward(self):
        activation = Activation_Linear()

        dvalues = np.array([[1, 2, 3], [4, 5, 6]])
        activation.backward(dvalues)

        # The backward pass of a linear activation function simply passes the gradient
        expected_dinputs = dvalues

        np.testing.assert_array_equal(activation.dinputs, expected_dinputs, err_msg="Backward pass failed.")

    def test_predictions(self):
        activation = Activation_Linear()

        outputs = np.array([[1, 2, 3], [4, 5, 6]])
        predictions = activation.predictions(outputs)

        # Predictions for a linear activation function should return the outputs as they are
        expected_predictions = outputs

        np.testing.assert_array_equal(predictions, expected_predictions, err_msg="Predictions failed.")
