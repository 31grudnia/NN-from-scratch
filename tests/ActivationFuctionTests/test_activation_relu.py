import numpy as np
import unittest

from . import Activation_Relu


class TestActivationRelu(unittest.TestCase):

    def test_forward(self):
        activation = Activation_Relu()
        
        # Test 1: Basic positive and negative input
        inputs = np.array([[1, -2, 3], [-1, 2, -3]])
        expected_output = np.array([[1, 0, 3], [0, 2, 0]])
        activation.forward(inputs)
        np.testing.assert_array_equal(activation.output, expected_output, err_msg="Relu Forward pass failed. Test 1")
        
        # Test 2: All negative input
        inputs = np.array([[-1, -2, -3], [-4, -5, -6]])
        expected_output = np.zeros_like(inputs)
        activation.forward(inputs)
        np.testing.assert_array_equal(activation.output, expected_output, err_msg="Relu Forward pass failed. Test 2")

        # Test 3: Mixed positive, negative, and zero input
        inputs = np.array([[0, 1, -1], [2, -2, 0]])
        expected_output = np.array([[0, 1, 0], [2, 0, 0]])
        activation.forward(inputs)
        np.testing.assert_array_equal(activation.output, expected_output, err_msg="Relu Forward pass failed. Test 3")

    def test_backward(self):
        activation = Activation_Relu()
        
        # Test 1: Basic backward pass
        inputs = np.array([[1, -2, 3], [-1, 2, -3]])
        dvalues = np.array([[1, 2, 3], [4, 5, 6]])
        expected_dinputs = np.array([[1, 0, 3], [0, 5, 0]])
        activation.forward(inputs)
        activation.backward(dvalues)
        np.testing.assert_array_equal(activation.dinputs, expected_dinputs, err_msg="Relu Backward pass failed. Test 1")
        
        # Test 2: All inputs are positive
        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        dvalues = np.array([[1, 2, 3], [4, 5, 6]])
        expected_dinputs = dvalues  # No zeroing out needed
        activation.forward(inputs)
        activation.backward(dvalues)
        np.testing.assert_array_equal(activation.dinputs, expected_dinputs, err_msg="Relu Backward pass failed. Test 2")

        # Test 3: Some inputs are zero
        inputs = np.array([[0, 1, -1], [2, 0, -2]])
        dvalues = np.array([[7, 8, 9], [10, 11, 12]])
        expected_dinputs = np.array([[0, 8, 0], [10, 0, 0]])
        activation.forward(inputs)
        activation.backward(dvalues)
        np.testing.assert_array_equal(activation.dinputs, expected_dinputs, err_msg="Relu Backward pass failed. Test 3")

    def test_predictions(self):
        activation = Activation_Relu()
        outputs = np.array([[1, 0, 3], [0, 2, 0]])
        predictions = activation.predictions(outputs)
        np.testing.assert_array_equal(predictions, outputs, err_msg="Relu Predictions func failed")
