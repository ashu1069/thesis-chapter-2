import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('TFT')
from modules import AddNorm

class TestAddNorm(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.skip_size = 8
        self.trainable_add = True
        self.model = AddNorm(self.input_size, self.skip_size, self.trainable_add)

    def test_initialization(self):
        model = AddNorm(self.input_size, self.skip_size, self.trainable_add)
        self.assertIsNotNone(model.norm)
        self.assertEqual(model.input_size, self.input_size)
        self.assertEqual(model.skip_size, self.skip_size)
        self.assertTrue(hasattr(model, 'resample'))
        self.assertTrue(hasattr(model, 'mask'))
        self.assertTrue(hasattr(model, 'gate'))

    def test_forward_output_shape(self):
        input_tensor = torch.randn(5, self.input_size)  # Batch size of 5
        skip_tensor = torch.randn(5, self.skip_size)    # Skip tensor with skip_size
        output_tensor = self.model(input_tensor, skip_tensor)
        expected_output_shape = (5, self.input_size)
        self.assertEqual(output_tensor.shape, expected_output_shape)

    def test_resampling(self):
        model = AddNorm(self.input_size, self.skip_size, self.trainable_add)
        input_tensor = torch.randn(5, self.input_size)
        skip_tensor = torch.randn(5, self.skip_size)
        output_tensor = model(input_tensor, skip_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)

    def test_trainable_add(self):
        model = AddNorm(self.input_size, self.skip_size, self.trainable_add)
        input_tensor = torch.randn(5, self.input_size)
        skip_tensor = torch.randn(5, self.skip_size)
        # Check if the mask is being used correctly
        with torch.no_grad():
            model.mask.fill_(1.0)
        output_tensor = model(input_tensor, skip_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)

if __name__ == '__main__':
    unittest.main()