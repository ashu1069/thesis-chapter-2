import unittest
import torch
import pandas as pd
import sys
sys.path.append('TFT')
from modules import GatedResidualNetwork

def load_static_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

class TestGRN(unittest.TestCase):
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

    def test_grn_initialization(self):
        grn = GatedResidualNetwork(input_size=1, hidden_size=8, output_size=1, dropout=0.1, context_size=6)
        self.assertIsInstance(grn, GatedResidualNetwork)
        self.assertEqual(grn.input_size, 1)
        self.assertEqual(grn.hidden_size, 8)
        self.assertEqual(grn.output_size, 1)
        self.assertEqual(grn.dropout, 0.1)
        self.assertEqual(grn.context_size, 6)

    def test_grn_forward_pass(self):
        grn = GatedResidualNetwork(input_size=1, hidden_size=8, output_size=1, dropout=0.1, context_size=6)
        batch_size = 32
        x = torch.randn(batch_size, 1)
        context = torch.randn(batch_size, 6)
        
        output = grn(x, context)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_grn_without_context(self):
        grn = GatedResidualNetwork(input_size=1, hidden_size=8, output_size=1, dropout=0.1, context_size=None)
        batch_size = 32
        x = torch.randn(batch_size, 1)
        
        output = grn(x)
        self.assertEqual(output.shape, (batch_size, 1))

    def test_load_static_data(self):
        static_data = load_static_data(self.file_paths)
        self.assertIsInstance(static_data, pd.DataFrame)
        for var in self.static_var_list:
            self.assertIn(var, static_data.columns)

    def test_grn_with_static_data(self):
        static_data = load_static_data(self.file_paths)
        
        batch_size = len(static_data)
        inputs = {var: torch.tensor(static_data[var].values).float().view(batch_size, -1) for var in self.static_var_list}
        
        grns = {var: GatedResidualNetwork(input_size=1, hidden_size=8, output_size=1, dropout=0.1, context_size=6) for var in self.static_var_list}
        
        context = torch.randn(batch_size, 6)
        outputs = {}
        
        for var, grn in grns.items():
            outputs[var] = grn(inputs[var], context)
            self.assertEqual(outputs[var].shape, (batch_size, 1))

    def test_grn_output_range(self):
        static_data = load_static_data(self.file_paths)
        
        batch_size = len(static_data)
        inputs = {var: torch.tensor(static_data[var].values).float().view(batch_size, -1) for var in self.static_var_list}
        
        grns = {var: GatedResidualNetwork(input_size=1, hidden_size=8, output_size=1, dropout=0.1, context_size=6) for var in self.static_var_list}
        
        context = torch.randn(batch_size, 6)
        
        for var, grn in grns.items():
            output = grn(inputs[var], context)
            self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1), f"Output for {var} is out of expected range")

if __name__ == '__main__':
    unittest.main()