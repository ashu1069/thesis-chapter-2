import unittest
import torch
import pandas as pd
import sys
import math
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('TFT')
from modules import ScaledDotProductAttention

class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 4
        self.dim = 8
        self.dropout = 0.1
        self.model = ScaledDotProductAttention(dropout=self.dropout, scale=True)

    def test_initialization(self):
        model = ScaledDotProductAttention(dropout=self.dropout, scale=True)
        self.assertIsNotNone(model.softmax)
        self.assertTrue(model.scale)
        self.assertIsInstance(model.dropout, nn.Dropout)

    def test_forward_output_shape(self):
        q = torch.randn(self.batch_size, self.seq_len, self.dim)
        k = torch.randn(self.batch_size, self.seq_len, self.dim)
        v = torch.randn(self.batch_size, self.seq_len, self.dim)
        output, attn = self.model(q, k, v)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
        self.assertEqual(attn.shape, (self.batch_size, self.seq_len, self.seq_len))

    def test_scaling(self):
        q = torch.randn(self.batch_size, self.seq_len, self.dim)
        k = torch.randn(self.batch_size, self.seq_len, self.dim)
        v = torch.randn(self.batch_size, self.seq_len, self.dim)
        model = ScaledDotProductAttention(dropout=self.dropout, scale=True)
        attn = torch.bmm(q, k.permute(0, 2, 1))
        scaled_attn = attn / math.sqrt(self.dim)
        model_output, model_attn = model(q, k, v)
        self.assertTrue(torch.allclose(model_attn, F.softmax(scaled_attn, dim=2), atol=1e-5))

    def test_masking(self):
        q = torch.randn(self.batch_size, self.seq_len, self.dim)
        k = torch.randn(self.batch_size, self.seq_len, self.dim)
        v = torch.randn(self.batch_size, self.seq_len, self.dim)
        mask = torch.ones(self.batch_size, self.seq_len, self.seq_len).bool()
        mask[:, :, -1] = 0
        output, attn = self.model(q, k, v, mask=mask)
        self.assertTrue(torch.all(attn[:, :, -1] < 1e-5))

    def test_dropout(self):
        q = torch.randn(self.batch_size, self.seq_len, self.dim)
        k = torch.randn(self.batch_size, self.seq_len, self.dim)
        v = torch.randn(self.batch_size, self.seq_len, self.dim)
        model = ScaledDotProductAttention(dropout=self.dropout, scale=True)
        model.eval()  # Disable dropout for consistency
        output, attn = model(q, k, v)
        model.train()  # Enable dropout
        dropout_output, dropout_attn = model(q, k, v)
        self.assertNotEqual(attn, dropout_attn)

if __name__ == '__main__':
    unittest.main()