import argparse
import os
import sys
import pandas as pd

# Ensure local module imports work when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from simulate import simulate_sequences
from methods import train_predict_lag, train_predict_tinygru


def main():
    parser = argparse.ArgumentParser(description='Clean scorer for vaccine proposals')
    parser.add_argument('--method', choices=['ridge', 'gbr', 'tinygru'], required=True)
    parser.add_argument('--num_proposals', type=int, default=400)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out_dir', type=str, default='experiments_outputs')
    args = parser.parse_args()

    known, static, targets = simulate_sequences(args.num_proposals, args.T, args.seed)

    if args.method in ['ridge', 'gbr']:
        dfX, obj_scores, equity, final, metrics = train_predict_lag(known, static, targets, model=args.method, seed=args.seed)
    else:
        dfX, obj_scores, equity, final = train_predict_tinygru(known, static, targets, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out = pd.DataFrame({'proposal_id': [f'P{i:04d}' for i in range(args.num_proposals)]})
    for k in obj_scores.columns:
        out[f'Score_{k}'] = obj_scores[k]
    out['Score_equity'] = equity
    out['Score_final'] = final

    csv_path = os.path.join(args.out_dir, f'scores_{args.method}.csv')
    out.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path}')


if __name__ == '__main__':
    main()


