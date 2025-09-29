Clean experiments
=================

Run minimal, self-contained baselines to score vaccine proposals.

Commands
--------

Ridge (very small data):

```
python3 experiments/score.py --method ridge --num_proposals 400 --T 5 --seed 11 --out_dir experiments_outputs
```

Gradient Boosting (mild nonlinearity):

```
python3 experiments/score.py --method gbr --num_proposals 400 --T 5 --seed 11 --out_dir experiments_outputs_gbr
```

TinyGRU + attention (sequential patterns):

```
python3 experiments/score.py --method tinygru --num_proposals 400 --T 5 --seed 21 --out_dir experiments_outputs_tinygru
```

Outputs
-------

- CSV: `scores_<method>.csv` with per-objective scores, equity, and final 0–100 score.

Notes
-----

- No target leakage: methods use only known + static features.
- Objective weights: 30/30/25/15 as specified; equity multiplies the weighted sum.

