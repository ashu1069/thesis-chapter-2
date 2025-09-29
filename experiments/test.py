import argparse
import os
import json
import numpy as np
import pandas as pd
import torch

from experiments.simulate import simulate_sequences
from experiments.methods import engineer_lag_features, TinyGRU, invert_to_scores_df
from experiments.methods import equity_from_static_df, finalize_scores
from experiments.config import FeatureConfig as C
from experiments.utils_io import load_sklearn_model, load_torch_state


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints and score proposals to CSV')
    parser.add_argument('--method', choices=['ridge', 'gbr', 'tinygru'], required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--num_proposals', type=int, default=400)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out_csv', type=str, default='scores.csv')
    args = parser.parse_args()

    known, static, targets = simulate_sequences(args.num_proposals, args.T, args.seed)

    if args.method in ['ridge', 'gbr']:
        with open(os.path.join(args.ckpt_dir, f'{args.method}_feature_cols.json')) as f:
            feature_cols = json.load(f)
        dfX = engineer_lag_features(known, static)
        X = dfX[feature_cols].values

        preds = {}
        for obj in C.OBJECTIVES:
            m = load_sklearn_model(os.path.join(args.ckpt_dir, f'{args.method}_{obj.replace(" ", "_")}.joblib'))
            preds[C.OBJECTIVES[obj]] = m.predict(X)
        preds_df = pd.DataFrame(preds)
        obj_scores = invert_to_scores_df(preds_df)
        equity = equity_from_static_df(dfX)
        final = finalize_scores(obj_scores, equity)

    else:
        state = load_torch_state(os.path.join(args.ckpt_dir, 'tinygru_state.pt'))
        model = TinyGRU(known_dim=known.shape[2], static_dim=static.shape[1], hidden=32)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(known, dtype=torch.float32), torch.tensor(static, dtype=torch.float32)).numpy()
        mins = y_pred.min(axis=0)
        maxs = y_pred.max(axis=0)
        scores = 1.0 - (y_pred - mins) / (maxs - mins + 1e-8)
        obj_scores = pd.DataFrame({
            'Maximize Health Impact': scores[:, 0],
            'Maximize Value for Money': scores[:, 1],
            'Reinforce Financial Sustainability': scores[:, 2],
            'Support Countries with the Greatest Needs': scores[:, 3],
        })
        dfX = engineer_lag_features(known, static)
        equity = equity_from_static_df(dfX)
        final = finalize_scores(obj_scores, equity)

    out = pd.DataFrame({'proposal_id': [f'P{i:04d}' for i in range(args.num_proposals)]})
    for k in obj_scores.columns:
        out[f'Score_{k}'] = obj_scores[k]
    out['Score_equity'] = equity
    out['Score_final'] = final
    out.to_csv(args.out_csv, index=False)
    print(f'Saved: {args.out_csv}')


if __name__ == '__main__':
    main()


