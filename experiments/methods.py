import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from config import FeatureConfig as C


def engineer_lag_features(known: np.ndarray, static: np.ndarray) -> pd.DataFrame:
    # known: [N, T, K], static: [N, S]
    N, T, K = known.shape
    cols = {}
    for s_idx, s_name in enumerate(C.STATIC):
        cols[f'static__{s_name}'] = static[:, s_idx]
    for k_idx, k_name in enumerate(C.KNOWN):
        series = known[:, :, k_idx]
        cols[f'known_last__{k_name}'] = series[:, -1]
        cols[f'known_mean__{k_name}'] = series.mean(axis=1)
        cols[f'known_std__{k_name}'] = series.std(axis=1)
        cols[f'known_slope__{k_name}'] = series[:, -1] - series[:, 0]
    return pd.DataFrame(cols)


def invert_to_scores_df(target_df: pd.DataFrame) -> pd.DataFrame:
    def inv(series: pd.Series) -> pd.Series:
        s = (series - series.min()) / (series.max() - series.min() + 1e-8)
        return 1.0 - s
    return pd.DataFrame({
        'Maximize Health Impact': inv(target_df[C.OBJECTIVES['Maximize Health Impact']]),
        'Maximize Value for Money': inv(target_df[C.OBJECTIVES['Maximize Value for Money']]),
        'Reinforce Financial Sustainability': inv(target_df[C.OBJECTIVES['Reinforce Financial Sustainability']]),
        'Support Countries with the Greatest Needs': inv(target_df[C.OBJECTIVES['Support Countries with the Greatest Needs']]),
    }, index=target_df.index)


def equity_from_static_df(df: pd.DataFrame) -> pd.Series:
    def minmax(s: pd.Series) -> pd.Series:
        return (s - s.min()) / (s.max() - s.min() + 1e-8)
    get = lambda name: df.get(f'static__{name}', None)
    parts = []
    p = get('Socio_economic_Poverty_Rates')
    g = get('Socio_economic_Gini_Index')
    b = get('Healthcare_Index_Bed_availability_per_capita')
    s = get('Political_Stability_Index')
    c = get('Security_and_Conflict_Index')
    if p is not None: parts.append(minmax(p))
    if g is not None: parts.append(minmax(g))
    if c is not None: parts.append(minmax(c))
    if b is not None: parts.append(1.0 - minmax(b))
    if s is not None: parts.append(1.0 - minmax(s))
    if not parts:
        return pd.Series(1.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)


def finalize_scores(obj_scores: pd.DataFrame, equity: pd.Series) -> pd.Series:
    weighted = sum(C.FINAL_WEIGHTS[k] * obj_scores[k].clip(0, 1) for k in C.FINAL_WEIGHTS)
    return (weighted * equity.clip(0, 1)) * 100.0


def train_predict_lag(known: np.ndarray, static: np.ndarray, targets: np.ndarray, model: str = 'ridge', seed: int = 123):
    dfX = engineer_lag_features(known, static)
    dfY = pd.DataFrame({
        C.OBJECTIVES['Maximize Health Impact']: targets[:, 0],
        C.OBJECTIVES['Maximize Value for Money']: targets[:, 1],
        C.OBJECTIVES['Reinforce Financial Sustainability']: targets[:, 2],
        C.OBJECTIVES['Support Countries with the Greatest Needs']: targets[:, 3],
    })
    feature_cols = list(dfX.columns)
    X = dfX.values

    models = {}
    metrics = {}
    X_train, X_val, idx_train, idx_val = train_test_split(X, np.arange(len(dfX)), test_size=0.25, random_state=seed)

    for obj_name, target_col in C.OBJECTIVES.items():
        y = dfY[target_col].values
        y_train, y_val = y[idx_train], y[idx_val]
        if model == 'ridge':
            m = Ridge(alpha=1.0, random_state=seed)
        else:
            m = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, learning_rate=0.05)
        m.fit(X_train, y_train)
        yp = m.predict(X_val)
        metrics[obj_name] = {'r2': float(r2_score(y_val, yp)), 'rmse': float(np.sqrt(mean_squared_error(y_val, yp)))}
        models[obj_name] = m

    # Predict all
    preds = {}
    for obj_name, m in models.items():
        preds[obj_name] = m.predict(dfX.values)
    preds_df = pd.DataFrame({
        C.OBJECTIVES['Maximize Health Impact']: preds['Maximize Health Impact'],
        C.OBJECTIVES['Maximize Value for Money']: preds['Maximize Value for Money'],
        C.OBJECTIVES['Reinforce Financial Sustainability']: preds['Reinforce Financial Sustainability'],
        C.OBJECTIVES['Support Countries with the Greatest Needs']: preds['Support Countries with the Greatest Needs'],
    })
    obj_scores = invert_to_scores_df(preds_df)
    equity = equity_from_static_df(dfX)
    final = finalize_scores(obj_scores, equity)
    return dfX, obj_scores, equity, final, metrics


class TinyGRU(nn.Module):
    def __init__(self, known_dim: int, static_dim: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_size=known_dim, hidden_size=hidden, batch_first=True)
        self.attn_vec = nn.Linear(hidden, hidden, bias=False)
        self.static_fc = nn.Sequential(nn.Linear(static_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 4))

    def forward(self, known, static):
        h_seq, h_last = self.gru(known)
        q = self.attn_vec(h_last[-1])
        attn_scores = torch.bmm(h_seq, q.unsqueeze(-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), h_seq).squeeze(1)
        s = self.static_fc(static)
        x = torch.cat([context, s], dim=1)
        return self.head(x)


def train_predict_tinygru(known: np.ndarray, static: np.ndarray, targets: np.ndarray, seed: int = 123):
    device = torch.device('cpu')
    N = known.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(0.75 * N)
    tr, va = idx[:split], idx[split:]

    Xk_tr = torch.tensor(known[tr], dtype=torch.float32)
    Xs_tr = torch.tensor(static[tr], dtype=torch.float32)
    y_tr = torch.tensor(targets[tr], dtype=torch.float32)
    Xk_va = torch.tensor(known[va], dtype=torch.float32)
    Xs_va = torch.tensor(static[va], dtype=torch.float32)
    y_va = torch.tensor(targets[va], dtype=torch.float32)

    model = TinyGRU(known_dim=known.shape[2], static_dim=static.shape[1], hidden=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best = float('inf')
    bad = 0
    for epoch in range(60):
        model.train()
        opt.zero_grad()
        yp = model(Xk_tr, Xs_tr)
        loss = loss_fn(yp, y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xk_va, Xs_va), y_va).item()
        if val < best - 1e-4:
            best = val
            bad = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        y_pred = model(torch.tensor(known, dtype=torch.float32), torch.tensor(static, dtype=torch.float32))

    # Convert predicted targets to objective scores
    y = y_pred.numpy()
    mins = y.min(axis=0)
    maxs = y.max(axis=0)
    norm = (y - mins) / (maxs - mins + 1e-8)
    scores = 1.0 - norm

    dfX = engineer_lag_features(known, static)
    obj_scores = pd.DataFrame({
        'Maximize Health Impact': scores[:, 0],
        'Maximize Value for Money': scores[:, 1],
        'Reinforce Financial Sustainability': scores[:, 2],
        'Support Countries with the Greatest Needs': scores[:, 3],
    })
    equity = equity_from_static_df(dfX)
    final = finalize_scores(obj_scores, equity)
    return dfX, obj_scores, equity, final


