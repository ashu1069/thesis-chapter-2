# vaccine_prioritization
Data Fusion Pipeline for Global Vaccine Prioritization
Quick start
-----------

Install deps:

```
pip install -r requirements.txt
```

Clean experiments (data-efficient TFT replacements):

```
# Ridge baseline
python3 experiments/score.py --method ridge --num_proposals 400 --T 5 --seed 11 --out_dir experiments_outputs

# TinyGRU + attention baseline
python3 experiments/score.py --method tinygru --num_proposals 400 --T 5 --seed 21 --out_dir experiments_outputs_tinygru
```

Training and testing with checkpoints:

```
# Train and save checkpoints
python3 experiments/train.py --method ridge --ckpt_dir checkpoints/ridge
python3 experiments/train.py --method tinygru --ckpt_dir checkpoints/tinygru

# Test from checkpoints
python3 experiments/test.py --method ridge --ckpt_dir checkpoints/ridge --out_csv ridge_scores.csv
python3 experiments/test.py --method tinygru --ckpt_dir checkpoints/tinygru --out_csv tinygru_scores.csv
```

TFT (full model): see `train.py` / `train_full_features.py`.