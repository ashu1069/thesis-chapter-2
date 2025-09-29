import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import torch
from torch.utils.data import Dataset

class DiseaseDataset(Dataset):
    def __init__(self, static_file, known_file, unknown_file, config):
        """
        Args:
            static_file (string): Path to the CSV file with static data.
            known_file (string): Path to the CSV file with time-dependent known data.
            unknown_file (string): Path to the CSV file with time-dependent unknown data.
            config (DataConfig): Configuration for variable lists.
        """
        self.static_data = pd.read_csv(static_file)
        self.known_data = pd.read_csv(known_file)
        self.unknown_data = pd.read_csv(unknown_file)
        
        # Extract variable lists from config
        self.static_vars = config.STATIC_VAR_LIST
        self.known_vars = config.TIME_KNOWN_VAR_LIST
        self.unknown_vars = config.TIME_UNKNOWN_VAR_LIST

        # Add these new lines to store target variables
        self.target_columns = [
            'Frequency_of_outbreaks',
            'Magnitude_of_outbreaks_Deaths',
            'Magnitude_of_outbreaks_Infected',
            'Magnitude_of_outbreaks_Severity_Index'
        ]

        # Check for NaNs in original data
        assert not self.known_data[self.known_vars].isnull().values.any(), "NaN in known data"
        if self.static_vars:
            assert not self.static_data[self.static_vars].isnull().values.any(), "NaN in static data"
        assert not self.unknown_data[self.target_columns].isnull().values.any(), "NaN in target data"

        # Normalize known features
        self.known_means = self.known_data[self.known_vars].mean()
        self.known_stds = self.known_data[self.known_vars].std().replace(0, 1)
        print("Known means:\n", self.known_means)
        print("Known stds:\n", self.known_stds)
        self.known_data[self.known_vars] = (self.known_data[self.known_vars] - self.known_means) / self.known_stds
        self.known_data[self.known_vars] = self.known_data[self.known_vars].fillna(0)

        # Normalize static features
        if self.static_vars:
            self.static_means = self.static_data[self.static_vars].mean()
            self.static_stds = self.static_data[self.static_vars].std().fillna(1).replace(0, 1)
            print("Static means:\n", self.static_means)
            print("Static stds:\n", self.static_stds)
            self.static_data[self.static_vars] = (self.static_data[self.static_vars] - self.static_means) / self.static_stds
            self.static_data[self.static_vars] = self.static_data[self.static_vars].fillna(0)
        else:
            self.static_means = None
            self.static_stds = None

        # Normalize targets
        self.target_means = self.unknown_data[self.target_columns].mean()
        self.target_stds = self.unknown_data[self.target_columns].std().replace(0, 1)
        print("Target means:\n", self.target_means)
        print("Target stds:\n", self.target_stds)
        self.unknown_data[self.target_columns] = (self.unknown_data[self.target_columns] - self.target_means) / self.target_stds
        self.unknown_data[self.target_columns] = self.unknown_data[self.target_columns].fillna(0)

    def __len__(self):
        # Return number of possible windows of length 5
        return min(len(self.known_data), len(self.unknown_data), len(self.static_data)) - 5 + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Static data: repeat the same static row for the window
        static_sample = self.static_data[self.static_vars].iloc[idx].values
        static_tensor = torch.tensor(static_sample, dtype=torch.float32).view(1, -1)
        static_tensor = static_tensor.repeat(5, 1)  # [5, static_features]

        # Temporal data: get a window of 5 time steps
        known_window = self.known_data[self.known_vars].iloc[idx:idx+5].values
        known_tensor = torch.tensor(known_window, dtype=torch.float32)  # [5, known_features]
        
        unknown_window = self.unknown_data[self.unknown_vars].iloc[idx:idx+5].values
        unknown_tensor = torch.tensor(unknown_window, dtype=torch.float32)  # [5, unknown_features]

        sample = {
            'static': static_tensor,   # [5, static_features]
            'known': known_tensor,     # [5, known_features]
            'unknown': unknown_tensor  # [5, unknown_features]
        }

        # Targets: use the last time step in the window
        targets = torch.tensor(self.unknown_data[self.target_columns].iloc[idx+4].values, dtype=torch.float32)
        
        # Denormalize targets for training (model outputs raw values)
        targets_denorm = targets * self.target_stds.values + self.target_means.values
        targets_denorm = targets_denorm.clone().detach().float()

        return {
            'inputs': sample,
            'targets': targets_denorm
        }
