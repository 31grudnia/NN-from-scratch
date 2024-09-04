import numpy as np
import unittest

from . import Loss_CategoricalCrossentropy


class TestLossCategoricalCrossentropy(unittest.TestCase):

    def setUp(self):
        self.loss_function = Loss_CategoricalCrossentropy()

    def test_forward_with_categorical_labels(self):
        y_pred = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
        y_true = np.array([0, 1, 1])

        expected_loss = -np.log([0.7, 0.5, 0.9])
        actual_loss = self.loss_function.forwad(y_pred, y_true)
        
        np.testing.assert_array_almost_equal(actual_loss, expected_loss, decimal=7)

    def test_forward_with_one_hot_labels(self):
        y_pred = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0]])

        expected_loss = -np.log([0.7, 0.5, 0.9])
        actual_loss = self.loss_function.forwad(y_pred, y_true)
        
        np.testing.assert_array_almost_equal(actual_loss, expected_loss, decimal=7)

    def test_backward(self):
        y_pred = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])
        y_true = np.array([0, 1, 1])

        # Calculate backward pass
        self.loss_function.backward(y_pred, y_true)
        expected_dinputs = np.array([[-1/0.7, 0, 0],
                                     [0, -1/0.5, 0],
                                     [0, -1/0.9, 0]]) / 3

        np.testing.assert_array_almost_equal(self.loss_function.dinputs, expected_dinputs, decimal=7)

    def test_backward_with_one_hot_labels(self):
        y_pred = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.05, 0.93]])
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        # Calculate backward pass
        self.loss_function.backward(y_pred, y_true)
        expected_dinputs = np.array([[-1/0.7, 0, 0],
                                     [0, -1/0.5, 0],
                                     [0, 0, -1/0.93]]) / 3

        np.testing.assert_array_almost_equal(self.loss_function.dinputs, expected_dinputs, decimal=7)
