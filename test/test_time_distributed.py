import unittest
import torch
import torch.nn as nn
import sys
sys.path.append('TFT')
from modules import TimeDistributed, TimeDistributedInterpolation

class TestTimeDistributed(unittest.TestCase):
    def setUp(self):
        self.linear = nn.Linear(10, 5)
        self.batch_size = 32
        self.time_steps = 20
        self.input_size = 10
        self.output_size = 5

    def test_initialization(self):
        td = TimeDistributed(self.linear)
        self.assertIsInstance(td, TimeDistributed)
        self.assertEqual(td.module, self.linear)
        self.assertFalse(td.batch_first)

    def test_forward_2d_input(self):
        td = TimeDistributed(self.linear)
        x = torch.randn(self.batch_size, self.input_size)
        output = td(x)
        self.assertEqual(output.shape, (self.batch_size, self.output_size))

    def test_forward_3d_input_batch_first_false(self):
        td = TimeDistributed(self.linear, batch_first=False)
        x = torch.randn(self.time_steps, self.batch_size, self.input_size)
        output = td(x)
        self.assertEqual(output.shape, (self.time_steps, self.batch_size, self.output_size))

    def test_forward_3d_input_batch_first_true(self):
        td = TimeDistributed(self.linear, batch_first=True)
        x = torch.randn(self.batch_size, self.time_steps, self.input_size)
        output = td(x)
        self.assertEqual(output.shape, (self.batch_size, self.time_steps, self.output_size))

    def test_forward_4d_input(self):
        td = TimeDistributed(self.linear, batch_first=True)
        x = torch.randn(self.batch_size, self.time_steps, 2, self.input_size)
        output = td(x)
        self.assertEqual(output.shape, (self.batch_size, self.time_steps*2, self.output_size))

    def test_module_application(self):
        td = TimeDistributed(self.linear)
        x = torch.randn(self.time_steps, self.batch_size, self.input_size)
        output = td(x)
        
        # Check if the module is applied correctly
        manual_output = torch.stack([self.linear(x[i]) for i in range(self.time_steps)])
        self.assertTrue(torch.allclose(output, manual_output, atol=1e-6))

class TestTimeDistributedInterpolation(unittest.TestCase):

    def setUp(self):
        self.input_size = 4
        self.output_size = 8
        self.batch_size = 2
        self.timesteps = 3

    def test_interpolation_non_trainable(self):
        model = TimeDistributedInterpolation(output_size=self.output_size, batch_first=True, trainable=False)
        x = torch.randn(self.batch_size, self.timesteps, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.timesteps, self.output_size))

    def test_interpolation_trainable(self):
        model = TimeDistributedInterpolation(output_size=self.output_size, batch_first=True, trainable=True)
        x = torch.randn(self.batch_size, self.timesteps, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.timesteps, self.output_size))
        # Check if mask is being applied
        self.assertTrue(model.mask.requires_grad)

    def test_interpolation_non_trainable_batch_first_false(self):
        model = TimeDistributedInterpolation(output_size=self.output_size, batch_first=False, trainable=False)
        x = torch.randn(self.timesteps, self.batch_size, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.timesteps, self.batch_size, self.output_size))

    def test_interpolation_trainable_batch_first_false(self):
        model = TimeDistributedInterpolation(output_size=self.output_size, batch_first=False, trainable=True)
        x = torch.randn(self.timesteps, self.batch_size, self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.timesteps, self.batch_size, self.output_size))
        # Check if mask is being applied
        self.assertTrue(model.mask.requires_grad)

    def test_interpolation_2d_input(self):
        model = TimeDistributedInterpolation(output_size=self.output_size, trainable=False)
        x = torch.randn(self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.output_size,))

    def test_interpolation_2d_input_trainable(self):
        model = TimeDistributedInterpolation(output_size=self.output_size, trainable=True)
        x = torch.randn(self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.output_size,))
        # Check if mask is being applied
        self.assertTrue(model.mask.requires_grad)

if __name__ == '__main__':
    unittest.main()