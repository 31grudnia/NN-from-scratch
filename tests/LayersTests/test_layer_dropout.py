# import unittest
# import numpy as np

# from . import Layer_Dropout


# class TestLayerDropout(unittest.TestCase):

#     def test_dropout_training(self):
#         layer = Layer_Dropout(0.5)
#         inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
#         layer.forward(inputs, training=True)
        
#         # Check that some values are dropped (set to zero) and others are scaled
#         dropped_count = np.sum(layer.output == 0)
#         scaled_count = np.sum((layer.output == 2 * inputs) | (layer.output == 0))
        
#         self.assertGreater(dropped_count, 0, "Some values should be dropped (set to zero).")
#         self.assertEqual(scaled_count, inputs.size, "Remaining values should be scaled.")

#     def test_dropout_inference(self):
#         layer = Layer_Dropout(0.5)
#         inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
#         layer.forward(inputs, training=False)
        
#         # Ensure the output is the same as the input
#         np.testing.assert_array_equal(layer.output, inputs, "During inference, output should match inputs.")

#     def test_dropout_rate_one(self):
#         layer = Layer_Dropout(1.0)
#         inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
#         layer.forward(inputs, training=True)
        
#         # With rate=1.0, all inputs should be retained
#         np.testing.assert_array_equal(layer.output, inputs, "With rate=1.0, output should match inputs.")

#     def test_dropout_rate_zero(self):
#         layer = Layer_Dropout(0.0)
#         inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
#         layer.forward(inputs, training=True)
        
#         # With rate=0.0, all outputs should be zero
#         np.testing.assert_array_equal(layer.output, np.zeros_like(inputs), "With rate=0.0, all outputs should be zero.")
    
#     def test_backward(self):
#         layer = Layer_Dropout(0.5)
#         inputs = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
#         layer.forward(inputs, training=True)
        
#         dvalues = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
#         layer.backward(dvalues)
        
#         # Check if the gradients are properly masked
#         expected_dinputs = dvalues * layer.binary_mask
#         np.testing.assert_array_almost_equal(layer.dinputs, expected_dinputs, err_msg="Backward pass gradients are not masked properly.")
