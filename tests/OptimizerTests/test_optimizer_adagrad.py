import numpy as np
import unittest

from . import Optimizer_Adagrad


# Define a mock layer class to simulate a layer with weights and biases
class MockLayer:
    def __init__(self, weights, biases):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.dweights = np.zeros_like(weights)
        self.dbiases = np.zeros_like(biases)

class TestOptimizerAdagrad(unittest.TestCase):

    def test_pre_update_params_without_decay(self):
        optimizer = Optimizer_Adagrad(learning_rate=0.1, decay=0)
        optimizer.pre_update_params()
        self.assertEqual(optimizer.current_learning_rate, 0.1)

    def test_pre_update_params_with_decay(self):
        optimizer = Optimizer_Adagrad(learning_rate=0.1, decay=0.01)
        optimizer.iterations = 10
        optimizer.pre_update_params()
        expected_learning_rate = 0.1 * (1. / (1. + 0.01 * 10))
        self.assertAlmostEqual(optimizer.current_learning_rate, expected_learning_rate, places=7)

    def test_update_params_with_caching(self):
        optimizer = Optimizer_Adagrad(learning_rate=0.1)
        mock_layer = MockLayer([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2])
        mock_layer.dweights = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_layer.dbiases = np.array([1.0, 2.0])
        
        optimizer.update_params(mock_layer)
        
        # Expected weights and biases after first update with Adagrad
        expected_weights = np.array([[0.1, 0.2], [0.3, 0.4]]) - 0.1 * mock_layer.dweights / (np.sqrt(mock_layer.dweights**2) + optimizer.epsilon)
        expected_biases = np.array([0.1, 0.2]) - 0.1 * mock_layer.dbiases / (np.sqrt(mock_layer.dbiases**2) + optimizer.epsilon)
        
        np.testing.assert_array_almost_equal(mock_layer.weights, expected_weights)
        np.testing.assert_array_almost_equal(mock_layer.biases, expected_biases)

        # Verify that the cache has been updated correctly
        expected_weight_cache = np.array([[1.0, 4.0], [9.0, 16.0]])
        expected_bias_cache = np.array([1.0, 4.0])
        np.testing.assert_array_almost_equal(mock_layer.weight_cache, expected_weight_cache)
        np.testing.assert_array_almost_equal(mock_layer.bias_cache, expected_bias_cache)

    def test_post_update_params(self):
        optimizer = Optimizer_Adagrad()
        initial_iterations = optimizer.iterations
        optimizer.post_update_params()
        self.assertEqual(optimizer.iterations, initial_iterations + 1)
