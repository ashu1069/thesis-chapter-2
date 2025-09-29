import argparse
import os
import json
import numpy as np
import pandas as pd
import torch

from experiments.simulate import simulate_sequences
from experiments.methods import train_predict_lag, train_predict_tinygru, TinyGRU
from experiments.methods import engineer_lag_features
from experiments.config import FeatureConfig as C
from experiments.utils_io import save_sklearn_model, save_torch_state


def main():
    parser = argparse.ArgumentParser(description='Train baseline models and save checkpoints')
    parser.add_argument('--method', choices=['ridge', 'gbr', 'tinygru'], required=True)
    parser.add_argument('--num_proposals', type=int, default=2000)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    known, static, targets = simulate_sequences(args.num_proposals, args.T, args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    meta = {'method': args.method, 'T': args.T, 'seed': args.seed, 'num_proposals': args.num_proposals}

    if args.method in ['ridge', 'gbr']:
        # Train and keep fitted sklearn models per objective
        dfX, obj_scores, equity, final, metrics = train_predict_lag(known, static, targets, model=args.method, seed=args.seed)
        # Refit on full data for saving
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor

        feature_cols = list(dfX.columns)
        X = dfX.values
        Y = {
            'Maximize Health Impact': targets[:, 0],
            'Maximize Value for Money': targets[:, 1],
            'Reinforce Financial Sustainability': targets[:, 2],
            'Support Countries with the Greatest Needs': targets[:, 3],
        }
        models = {}
        for obj, y in Y.items():
            if args.method == 'ridge':
                m = Ridge(alpha=1.0, random_state=args.seed)
            else:
                m = GradientBoostingRegressor(random_state=args.seed, n_estimators=200, max_depth=3, learning_rate=0.05)
            m.fit(X, y)
            models[obj] = m
            save_sklearn_model(m, os.path.join(args.ckpt_dir, f'{args.method}_{obj.replace(" ", "_")}.joblib'))

        with open(os.path.join(args.ckpt_dir, f'{args.method}_feature_cols.json'), 'w') as f:
            json.dump(feature_cols, f)
        with open(os.path.join(args.ckpt_dir, f'{args.method}_meta.json'), 'w') as f:
            json.dump({'metrics': metrics, **meta}, f)

        print(f"Saved sklearn checkpoints to {args.ckpt_dir}")

    else:
        # Train TinyGRU and save state_dict
        # Quick training on whole dataset
        model = TinyGRU(known_dim=known.shape[2], static_dim=static.shape[1], hidden=32)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = torch.nn.MSELoss()
        Xk = torch.tensor(known, dtype=torch.float32)
        Xs = torch.tensor(static, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        best = float('inf')
        bad = 0
        for epoch in range(80):
            model.train()
            opt.zero_grad()
            yp = model(Xk, Xs)
            loss = loss_fn(yp, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if loss.item() < best - 1e-4:
                best = loss.item()
                bad = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break

        save_torch_state(best_state, os.path.join(args.ckpt_dir, 'tinygru_state.pt'))
        with open(os.path.join(args.ckpt_dir, 'tinygru_meta.json'), 'w') as f:
            json.dump(meta, f)
        print(f"Saved TinyGRU checkpoint to {args.ckpt_dir}")


if __name__ == '__main__':
    main()


