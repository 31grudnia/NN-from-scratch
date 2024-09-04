import numpy as np
import unittest

from . import Activation_Sigmoid


class TestActivationSigmoid(unittest.TestCase):

    def test_forward(self):
        activation = Activation_Sigmoid()

        inputs = np.array([[0, 2, -2], [1, -1, 0]])
        activation.forward(inputs)

        # Expected output calculated manually
        expected_output = 1 / (1 + np.exp(-inputs))

        np.testing.assert_allclose(activation.output, expected_output, rtol=1e-5, err_msg="Forward pass failed.")

    def test_backward(self):
        activation = Activation_Sigmoid()

        inputs = np.array([[0, 2, -2], [1, -1, 0]])
        dvalues = np.array([[1, 1, 1], [1, 1, 1]])

        # Perform forward pass to set activation.output
        activation.forward(inputs)
        activation.backward(dvalues)

        # Expected output for backward pass
        sigmoid_derivative = (1 - activation.output) * activation.output
        expected_dinputs = dvalues * sigmoid_derivative

        np.testing.assert_allclose(activation.dinputs, expected_dinputs, rtol=1e-5, err_msg="Backward pass failed.")

    def test_predictions(self):
        activation = Activation_Sigmoid()

        outputs = np.array([[0.7, 0.2, 0.8], [0.4, 0.5, 0.6]])
        predictions = activation.predictions(outputs)

        # Expected predictions
        expected_predictions = (outputs > 0.5) * 1

        np.testing.assert_array_equal(predictions, expected_predictions, err_msg="Predictions failed.")
