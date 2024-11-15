import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import torch
from torch.utils.data import Dataset

class DiseaseDataset(Dataset):
    def __init__(self, static_file, known_file, unknown_file, config, transform=None):
        """
        Args:
            static_file (string): Path to the CSV file with static data.
            known_file (string): Path to the CSV file with time-dependent known data.
            unknown_file (string): Path to the CSV file with time-dependent unknown data.
            config (DataConfig): Configuration for variable lists.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.static_data = pd.read_csv(static_file)
        self.known_data = pd.read_csv(known_file)
        self.unknown_data = pd.read_csv(unknown_file)
        self.transform = transform
        
        # Normalize categorical data
        self.static_data['Transmission_Mode'] = self.static_data['Transmission_Mode'].astype('category').cat.codes

        # Repeat static data to match the length of time-dependent data
        self.static_data = pd.concat([self.static_data]*len(self.known_data), ignore_index=True)

        # Extract variable lists from config
        self.static_vars = config.get_variable('STATIC_VAR_LIST')
        self.known_vars = config.get_variable('TIME_KNOWN_VAR_LIST')
        self.unknown_vars = config.get_variable('TIME_UNKNOWN_VAR_LIST')

        # Add these new lines to store target variables
        self.target_vars = {
            'health': config.get_variable('HEALTH_TARGET_VAR'),
            'value': config.get_variable('VALUE_TARGET_VAR'),
            'sustainability': config.get_variable('SUSTAINABILITY_TARGET_VAR'),
            'needs': config.get_variable('NEEDS_TARGET_VAR'),
            'equity': config.get_variable('EQUITY_TARGET_VAR')
        }

    def __len__(self):
        return len(self.known_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Ensure static data is 2D: [batch_size, features]
        static_sample = self.static_data[self.static_vars].iloc[idx].values
        static_tensor = torch.tensor(static_sample, dtype=torch.float).view(1, -1)

        # Ensure temporal data is 3D: [batch_size, time_steps, features]
        known_sample = self.known_data[self.known_vars].iloc[idx].values
        known_tensor = torch.tensor(known_sample, dtype=torch.float).view(1, -1, len(self.known_vars))

        unknown_sample = self.unknown_data[self.unknown_vars].iloc[idx].values
        unknown_tensor = torch.tensor(unknown_sample, dtype=torch.float).view(1, -1, len(self.unknown_vars))

        sample = {
            'static': static_tensor,
            'known': known_tensor,
            'unknown': unknown_tensor
        }

        if self.transform:
            sample = self.transform(sample)

        # Ensure targets are properly shaped
        targets = torch.tensor([
            self.unknown_data[self.target_vars['health']].iloc[idx],
            self.unknown_data[self.target_vars['value']].iloc[idx],
            self.unknown_data[self.target_vars['sustainability']].iloc[idx],
            self.unknown_data[self.target_vars['needs']].iloc[idx]
        ], dtype=torch.float)

        return {
            'inputs': sample,
            'targets': targets
        }

from vaccine_prioritization.utils.config import DataConfig, VaccineData

# Define the configuration
config = DataConfig(
    STATIC_VAR_LIST=VaccineData.STATIC_VAR_LIST,
    TIME_KNOWN_VAR_LIST=VaccineData.TIME_KNOWN_VAR_LIST,
    TIME_UNKNOWN_VAR_LIST=VaccineData.TIME_UNKNOWN_VAR_LIST
)

# Define file paths
static_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "static_data_COVID-19.csv")
known_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "time_dependent_known_data_COVID-19.csv")
unknown_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "time_dependent_unknown_data_COVID-19.csv")

# Create dataset
dataset = DiseaseDataset(static_file, known_file, unknown_file, config)

# Access a sample
sample = dataset[0]
print(sample)

# Use DataLoader to iterate through the dataset
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for i, batch in enumerate(dataloader):
    print(batch)
    break
