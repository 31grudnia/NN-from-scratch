import unittest
import numpy as np

from . import Optimizer_Adam


# Define a mock layer class to simulate a layer with weights and biases
class MockLayer:
    def __init__(self, weights, biases):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.dweights = np.zeros_like(weights)
        self.dbiases = np.zeros_like(biases)

class TestOptimizerAdam(unittest.TestCase):

    def test_pre_update_params_without_decay(self):
        optimizer = Optimizer_Adam(learning_rate=0.01, decay=0)
        optimizer.pre_update_params()
        self.assertEqual(optimizer.current_learning_rate, 0.01)

    def test_pre_update_params_with_decay(self):
        optimizer = Optimizer_Adam(learning_rate=0.01, decay=0.01)
        optimizer.iterations = 10
        optimizer.pre_update_params()
        expected_learning_rate = 0.01 * (1. / (1. + 0.01 * 10))
        self.assertAlmostEqual(optimizer.current_learning_rate, expected_learning_rate, places=7)

    def test_update_params_with_momentum_and_cache(self):
        optimizer = Optimizer_Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        mock_layer = MockLayer([[0.2, 0.4], [0.6, 0.8]], [0.2, 0.4])
        mock_layer.dweights = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_layer.dbiases = np.array([1.0, 2.0])

        optimizer.update_params(mock_layer)

        # Calculate expected weight and bias momentums and caches
        expected_weight_momentums = 0.9 * np.zeros_like(mock_layer.dweights) + 0.1 * mock_layer.dweights
        expected_bias_momentums = 0.9 * np.zeros_like(mock_layer.dbiases) + 0.1 * mock_layer.dbiases
        weight_momentums_corrected = expected_weight_momentums / (1 - 0.9 ** (optimizer.iterations + 1))
        bias_momentums_corrected = expected_bias_momentums / (1 - 0.9 ** (optimizer.iterations + 1))

        expected_weight_cache = 0.999 * np.zeros_like(mock_layer.dweights) + 0.001 * mock_layer.dweights**2
        expected_bias_cache = 0.999 * np.zeros_like(mock_layer.dbiases) + 0.001 * mock_layer.dbiases**2
        weight_cache_corrected = expected_weight_cache / (1 - 0.999 ** (optimizer.iterations + 1))
        bias_cache_corrected = expected_bias_cache / (1 - 0.999 ** (optimizer.iterations + 1))

        # Calculate expected weights and biases after update
        expected_weights = np.array([[0.2, 0.4], [0.6, 0.8]]) - \
            0.01 * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + 1e-7)
        expected_biases = np.array([0.2, 0.4]) - \
            0.01 * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + 1e-7)

        # Check if the layer's weights and biases were updated correctly
        np.testing.assert_array_almost_equal(mock_layer.weights, expected_weights, decimal=7)
        np.testing.assert_array_almost_equal(mock_layer.biases, expected_biases, decimal=7)

    def test_post_update_params(self):
        optimizer = Optimizer_Adam()
        initial_iterations = optimizer.iterations
        optimizer.post_update_params()
        self.assertEqual(optimizer.iterations, initial_iterations + 1)

