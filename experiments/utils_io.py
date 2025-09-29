import os
import joblib
import torch


def save_sklearn_model(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_sklearn_model(path: str):
    return joblib.load(path)


def save_torch_state(state_dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


def load_torch_state(path: str):
    return torch.load(path, map_location='cpu')


