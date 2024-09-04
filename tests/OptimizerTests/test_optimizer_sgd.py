import numpy as np
import unittest

from . import Optimizer_SGD


# Define a mock layer class to simulate a layer with weights and biases
class MockLayer:
    def __init__(self, weights, biases):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.dweights = np.zeros_like(weights)
        self.dbiases = np.zeros_like(biases)

class TestOptimizerSGD(unittest.TestCase):

    def test_pre_update_params_without_decay(self):
        optimizer = Optimizer_SGD(learning_rate=0.1, decay=0)
        optimizer.pre_update_params()
        self.assertEqual(optimizer.current_learning_rate, 0.1)

    def test_pre_update_params_with_decay(self):
        optimizer = Optimizer_SGD(learning_rate=0.1, decay=0.01)
        optimizer.iterations = 10
        optimizer.pre_update_params()
        expected_learning_rate = 0.1 * (1. / (1. + 0.01 * 10))
        self.assertAlmostEqual(optimizer.current_learning_rate, expected_learning_rate, places=7)

    def test_update_params_without_momentum(self):
        optimizer = Optimizer_SGD(learning_rate=0.1, momentum=0)
        mock_layer = MockLayer([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2])
        mock_layer.dweights = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_layer.dbiases = np.array([1.0, 2.0])
        
        optimizer.update_params(mock_layer)
        
        np.testing.assert_array_almost_equal(mock_layer.weights, np.array([[0.0, 0.0], [0.0, 0.0]]))
        np.testing.assert_array_almost_equal(mock_layer.biases, np.array([0.0, 0.0]))

    def test_update_params_with_momentum(self):
        optimizer = Optimizer_SGD(learning_rate=0.1, momentum=0.9)
        mock_layer = MockLayer([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2])
        mock_layer.dweights = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_layer.dbiases = np.array([1.0, 2.0])
        
        optimizer.update_params(mock_layer)
        
        np.testing.assert_array_almost_equal(mock_layer.weights, np.array([[0.0, 0.0], [0.0, 0.0]]))
        np.testing.assert_array_almost_equal(mock_layer.biases, np.array([0.0, 0.0]))

        # Second update to check momentum
        optimizer.update_params(mock_layer)
        
        expected_weights = np.array([[-0.19, -0.38], [-0.57, -0.76]])
        expected_biases = np.array([-0.19, -0.38])
        np.testing.assert_array_almost_equal(mock_layer.weights, expected_weights)
        np.testing.assert_array_almost_equal(mock_layer.biases, expected_biases)

    def test_post_update_params(self):
        optimizer = Optimizer_SGD()
        initial_iterations = optimizer.iterations
        optimizer.post_update_params()
        self.assertEqual(optimizer.iterations, initial_iterations + 1)