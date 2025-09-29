import os
import sys
import numpy as np
from config import FeatureConfig as C


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def simulate_sequences(num_proposals: int, T: int, seed: int = 123):
    set_seed(seed)
    S = len(C.STATIC)
    K = len(C.KNOWN)

    static = np.clip(np.random.beta(2, 2, size=(num_proposals, S)), 0, 1)

    known = np.zeros((num_proposals, T, K), dtype=np.float32)
    for n in range(num_proposals):
        for k in range(K):
            level = np.clip(np.random.normal(0.5, 0.1), 0, 1)
            for t in range(T):
                level = np.clip(0.7 * level + 0.3 * np.random.normal(0.5, 0.2), 0, 1)
                known[n, t, k] = level

    wk1 = np.random.uniform(-0.6, 0.6, size=K)
    ws = np.random.uniform(-0.4, 0.4, size=S)
    base = 0.5 + 0.4 * (known[:, -1, :] @ wk1) + 0.3 * (static @ ws)
    base = np.clip(base, -2, 2)

    targets = np.stack([
        np.clip(0.5 + 0.6 * np.tanh(base + np.random.normal(0, 0.1, size=num_proposals)), 0, 1),
        np.clip(0.5 + 0.6 * np.tanh(0.9 * base + np.random.normal(0, 0.1, size=num_proposals)), 0, 1),
        np.clip(0.5 + 0.6 * np.tanh(1.1 * base + 0.1 + np.random.normal(0, 0.1, size=num_proposals)), 0, 1),
        np.clip(0.5 + 0.6 * np.tanh(base + np.random.normal(0, 0.1, size=num_proposals)), 0, 1),
    ], axis=1).astype(np.float32)

    return known, static, targets


