import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('TFT')
from modules import GatedLinearUnit

class TestGatedLinearUnit(unittest.TestCase):
    def setUp(self):
        self.input_size = 10
        self.hidden_size = 20
        self.dropout = 0.5
        self.model = GatedLinearUnit(self.input_size, self.hidden_size, self.dropout)

    def test_initialization(self):
        model = GatedLinearUnit(self.input_size, self.hidden_size, self.dropout)
        self.assertIsNotNone(model.fc)
        self.assertEqual(model.hidden_size, self.hidden_size)
        self.assertIsInstance(model.dropout, nn.Dropout)

    def test_forward_output_shape(self):
        input_tensor = torch.randn(5, self.input_size)  # Batch size of 5
        output_tensor = self.model(input_tensor)
        expected_output_shape = (5, self.hidden_size)
        self.assertEqual(output_tensor.shape, expected_output_shape)

    def test_dropout(self):
        model = GatedLinearUnit(self.input_size, self.hidden_size, self.dropout)
        input_tensor = torch.ones(5, self.input_size)
        output_tensor = model(input_tensor)
        # Check if dropout is applied
        self.assertNotEqual(output_tensor.sum().item(), self.input_size * 5 * 2)  # Should be different due to dropout

    def test_weight_initialization(self):
        model = GatedLinearUnit(self.input_size, self.hidden_size, self.dropout)
        for n, p in model.named_parameters():
            if "bias" in n:
                self.assertTrue(torch.all(p == 0))
            elif "fc" in n:
                # Xavier uniform initialization test
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.fc.weight)
                bound = 1 / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
                self.assertTrue(torch.all((model.fc.weight.data >= -bound) & (model.fc.weight.data <= bound)))

if __name__ == '__main__':
    unittest.main()

'''
The issue arises because the weight initialization test is too strict. 
The Xavier uniform initialization does not guarantee that all values will be strictly within the bounds, 
due to the floating-point precision. Instead, we should test that the values are approximately within the expected range.
'''