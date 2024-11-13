import unittest
import torch
import pandas as pd
import sys
import os

# Adjust the path to include the directory containing your module
sys.path.append('TFT')
from modules import VariableSelectionNetwork, GatedResidualNetwork, ResampleNorm

def load_static_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

class TestVariableSelectionNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_paths = [
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_Chikungunya.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_COVID-19.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_Dengue.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_Group B Streptococcus.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_Chikungunya.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_Hepatitis-E.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_M-pox.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_Shigella.csv',
            '/home/stu12/s11/ak1825/hsel/static_data/static_data_TB.csv'
        ]
        cls.static_var_list = [
            'Endemic Potential R0', 'Endemic Potential CFR', 'Endemic Potential Duration',
            'Demography Urban-Rural Split', 'Demography Population Density', 'Environmental Index',
            'Communication Affordability', 'Communication Media Freedom', 'Communication Connectivity',
            'Socio-economic GDP per capita', 'Socio-economic Gini Index', 'Socio-economic Employment Rates',
            'Socio-economic Poverty Rates', 'Socio-economic Education Levels'
        ]
        cls.static_data = load_static_data(cls.file_paths)

    def test_vsn_initialization(self):
        input_sizes = {var: 1 for var in self.static_var_list}
        vsn = VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=len(self.static_var_list),
            input_embedding_flags={var: False for var in self.static_var_list},
            dropout=0.1,
            context_size=6
        )
        self.assertIsInstance(vsn, VariableSelectionNetwork)
        self.assertEqual(vsn.hidden_size, 14)
        self.assertEqual(vsn.dropout, 0.1)
        self.assertEqual(vsn.context_size, 6)

    def test_vsn_forward_pass(self):
        input_sizes = {var: 1 for var in self.static_var_list}
        vsn = VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=len(self.static_var_list),
            input_embedding_flags={var: False for var in self.static_var_list},
            dropout=0.1,
            context_size=6
        )
        batch_size = len(self.static_data)
        inputs = {var: torch.tensor(self.static_data[var].values).float().view(batch_size, 1) for var in self.static_var_list}
        context = torch.randn(batch_size, 6)
        
        outputs, sparse_weights = vsn(inputs, context)
        
        self.assertEqual(outputs.shape, (batch_size, len(self.static_var_list)))
        self.assertEqual(sparse_weights.shape, (batch_size, len(self.static_var_list), 1))

        # Print shapes for debugging
        print(f"Outputs shape: {outputs.shape}")
        print(f"Sparse weights shape: {sparse_weights.shape}")

    def test_vsn_output_range(self):
        input_sizes = {var: 1 for var in self.static_var_list}
        vsn = VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=len(self.static_var_list),
            input_embedding_flags={var: False for var in self.static_var_list},
            dropout=0.1,
            context_size=6
        )
        batch_size = len(self.static_data)
        inputs = {var: torch.tensor(self.static_data[var].values).float().view(batch_size, 1) for var in self.static_var_list}
        context = torch.randn(batch_size, 6)
        
        outputs, sparse_weights = vsn(inputs, context)
        
        self.assertTrue(torch.all(torch.isfinite(outputs)), "Outputs contain non-finite values")
        self.assertTrue(torch.all(sparse_weights >= 0) and torch.all(sparse_weights <= 1), "Sparse weights are out of expected range")

        # Print min and max values for debugging
        print(f"Outputs min: {outputs.min().item()}, max: {outputs.max().item()}")
        print(f"Sparse weights min: {sparse_weights.min().item()}, max: {sparse_weights.max().item()}")

    def test_vsn_single_input(self):
        input_sizes = {'single_var': 1}
        vsn = VariableSelectionNetwork(
            input_sizes=input_sizes,
            hidden_size=len(self.static_var_list),
            input_embedding_flags={'single_var': False},
            dropout=0.1,
            context_size=6
        )
        batch_size = len(self.static_data)
        inputs = {'single_var': torch.randn(batch_size, 1)}
        context = torch.randn(batch_size, 6)
        
        outputs, sparse_weights = vsn(inputs, context)
        
        self.assertEqual(outputs.shape, (batch_size, 14))
        self.assertEqual(sparse_weights.shape, (batch_size, 1, 1))

if __name__ == '__main__':
    unittest.main()