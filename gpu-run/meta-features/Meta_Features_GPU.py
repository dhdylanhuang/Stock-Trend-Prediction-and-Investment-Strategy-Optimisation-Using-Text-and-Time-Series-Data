#!/usr/bin/env python
# coding: utf-8

# # Meta Features Built on GPU

# ### Imports

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


import numpy as np

import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, matthews_corrcoef


# ### Paths for GPU 

# In[ ]:


import os
from pathlib import Path

PERSIST_ROOT = Path(os.environ.get('PERSIST_ROOT', '/mnt/primary'))
if not PERSIST_ROOT.exists():
    raise RuntimeError(f'Persistent storage not found at {PERSIST_ROOT}. Check mounts (df -h /mnt/primary).')

RUN_ROOT = Path(os.environ.get('RUN_ROOT', PERSIST_ROOT / 'meta-features'))
if not str(RUN_ROOT).startswith(str(PERSIST_ROOT)):
    print(f'WARNING: RUN_ROOT={RUN_ROOT} is not on persistent storage; forcing to {PERSIST_ROOT}/meta-features')
    RUN_ROOT = PERSIST_ROOT / 'meta-features'
RUN_ROOT.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(os.environ.get('DATA_PATH', RUN_ROOT / 'ta_nlp_sector.parquet'))
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', RUN_ROOT / 'results'))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
META_OUT_PATH = Path(os.environ.get('META_OUT_PATH', RUN_ROOT / 'master_df_20rf.parquet'))
META_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print('DATA_PATH:', DATA_PATH)
print('RESULTS_ROOT:', RESULTS_ROOT)
print('META_OUT_PATH:', META_OUT_PATH)


# ### Constants

# In[ ]:


L_CLASSIFICATION = 12
REFIT_INTERVAL_CLASSIFICATION = 20
W_BASE_CLASSIFICATION = 250
MIN_SEQ_CLASSIFICATION = 40

L_REGRESSION = 12     # default sequence length for meta regression (can be overridden per-model)
REFIT_INTERVAL = 20  # refit cadence in trading days
W_BASE = 250          # warm-up period before first predictions
MIN_SEQ = 40          # minimum sequences required to train

H_META = 1 # 1-day ahead meta target

# DataLoader tuning for 4 CPU cores (adjust if needed)
NUM_WORKERS_TRAIN = 1
NUM_WORKERS_EVAL = 1
PIN_MEMORY = True
PERSISTENT_WORKERS = True


# ### Features

# In[ ]:


# Feature sets (aligned with Benchmarking.ipynb)
feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'roll_ret_1d', 'roll_ret_5d', 'roll_ret_20d',
    'ema_12', 'ema_26', 'ema_50', 'macd_12_26_9', 'macdh_12_26_9',
    'macds_12_26_9', 'rsi_14', 'stochrsik_14_14_3_3', 'stochrsid_14_14_3_3',
    'atrr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'obv',
]

sentiment_columns = ['sentiment']

emotion_columns = [
    'emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy',
    'emotion_neutral', 'emotion_sadness', 'emotion_surprize',
    'emotion_anger_pct', 'emotion_disgust_pct', 'emotion_fear_pct',
    'emotion_joy_pct', 'emotion_neutral_pct', 'emotion_sadness_pct',
    'emotion_surprize_pct',
]

unified_emotion_columns = [
    'positive_emotion', 'negative_emotion', 'uncertainty_emotion',
    'positive_emotion_pct', 'negative_emotion_pct', 'uncertainty_emotion_pct',
]

stance_columns = ['stance_label', 'stance_score']

finbert_columns = ['finbert_label', 'finbert_score', 'finbert_up', 'finbert_down', 'finbert_neutral']

sector_columns = [
    'sector_open_mean', 'sector_high_mean', 'sector_low_mean', 'sector_close_mean',
    'sector_volume_mean', 'sector_ret_1d', 'sector_ret_5d', 'sector_ret_20d',
    'sector_range', 'sector_vol_20d',
    'ema_12_sector', 'ema_26_sector', 'ema_50_sector', 'macd_12_26_9_sector',
    'macdh_12_26_9_sector', 'macds_12_26_9_sector', 'rsi_14_sector',
    'sector_bb_upper', 'sector_bb_middle', 'sector_bb_lower',
    'market_close', 'sector_rel_strength', 'sector_dispersion_1d',
]

SAFE_FFILL_COLS = list(dict.fromkeys(feature_columns + sector_columns))

# Feature sets and groups
feature_sets = {
    'base': feature_columns,
    'sentiment': feature_columns + sentiment_columns,
    'emotion': feature_columns + emotion_columns,
    'unified_emotion': feature_columns + unified_emotion_columns,
    'finbert': feature_columns + finbert_columns,
    'all_nlp': feature_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    'sector': feature_columns + sector_columns,
    'sector_sentiment': feature_columns + sector_columns + sentiment_columns,
    'sector_emotion': feature_columns + sector_columns + emotion_columns,
    'sector_unified_emotion': feature_columns + sector_columns + unified_emotion_columns,
    'sector_finbert': feature_columns + sector_columns + finbert_columns,
    'sector_all_nlp': feature_columns + sector_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    # typo alias used in Benchmarking
    'sentinment': feature_columns + sentiment_columns,
}

feature_groups = {
    'base': ['base'],
    'nlp': ['sentiment', 'emotion', 'unified_emotion', 'finbert', 'all_nlp'],
    'sector': ['sector', 'sector_sentiment', 'sector_emotion', 'sector_unified_emotion', 'sector_finbert', 'sector_all_nlp'],
}

FEATURE_SET_ALIAS = {
    'sentinment': 'sentiment',
}


# ### Helper Functions

# In[ ]:


def _resolve_feature_set_name(name: str):
    name = FEATURE_SET_ALIAS.get(name, name)
    if name not in feature_sets:
        raise KeyError(f"Unknown feature set '{name}'. Available: {sorted(feature_sets.keys())}")
    return name


def _validate_feature_cols(feature_cols, df_cols):
    missing = [c for c in feature_cols if c not in df_cols]
    if missing:
        raise ValueError(f"Missing feature columns in master_df: {missing}")


def _get_param(params, key, default=None, cast=None):
    if key in params and pd.notna(params[key]):
        val = params[key]
    else:
        val = default
    return cast(val) if cast is not None and val is not None else val


def select_best_configs(task='classification', metric='mcc', models=('LSTM','BiLSTM','GRU','BiGRU')):
    base_dir = RESULTS_ROOT / 'benchmarking' / task
    if not base_dir.exists():
        base_dir = Path('results/benchmarking') / task
    if not base_dir.exists():
        raise FileNotFoundError(f"No results directory found: {base_dir}")

    # collect all H=1 results across feature groups
    rows = []
    for group_dir in sorted(base_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for model in models:
            results_path = group_dir / f"{model}_1H_results.csv"
            params_path = group_dir / f"{model}_1H_params.csv"
            if not results_path.exists() or not params_path.exists():
                continue

            df = pd.read_csv(results_path)
            if metric not in df.columns:
                continue
            score = float(df[metric].mean())

            params_row = pd.read_csv(params_path).iloc[0].to_dict()
            feature_set = params_row.get('params_feature_set', group)
            feature_set = _resolve_feature_set_name(str(feature_set))

            rows.append({
                'model': model,
                'feature_group': group,
                'feature_set': feature_set,
                'score': score,
                'params': params_row,
                'results_path': str(results_path),
                'params_path': str(params_path),
            })

    if not rows:
        raise FileNotFoundError(f"No H=1 results/params found under {base_dir}")

    # sort by metric desc and pick first occurrence per model
    rows = sorted(rows, key=lambda r: r['score'], reverse=True)
    best = {}
    for r in rows:
        if r['model'] not in best:
            best[r['model']] = {
                'feature_group': r['feature_group'],
                'feature_set': r['feature_set'],
                'mcc_mean': r['score'],
                'params': r['params'],
                'results_path': r['results_path'],
                'params_path': r['params_path'],
            }
        if len(best) == len(models):
            break

    missing = [m for m in models if m not in best]
    if missing:
        raise FileNotFoundError(f"Missing best configs for models: {missing}")

    return best


def ffill_only(df, cols):
    # forward-fill only stable features; leave sparse NLP as-is
    safe_cols = [c for c in cols if c in SAFE_FFILL_COLS]
    if not safe_cols:
        return df
    return df.assign(**{c: df[c].ffill() for c in safe_cols})


def fit_scaler_on_train(train_df, feature_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    return scaler


def transform_df(df, scaler, feature_cols):
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols].values)
    return out


def build_sequences_meta(df, feature_cols, target_col, seq_len):
    df = df.sort_values('date').reset_index(drop=True)
    X_vals = df[feature_cols].values
    y_vals = df[target_col].values
    X_list, y_list, idx_list = [], [], []
    for i in range(seq_len - 1, len(df)):
        window_X = X_vals[i-seq_len+1:i+1]  # window ending at time t
        y = y_vals[i]                       # forward return t -> t+1
        if np.isnan(window_X).any() or pd.isna(y):
            continue
        X_list.append(window_X)
        y_list.append(y)
        idx_list.append(i)
    return np.array(X_list), np.array(y_list), idx_list


# ## Load Dataset

# In[6]:


master_df = pd.read_parquet(DATA_PATH)
master_df.reset_index(drop=True, inplace=True)

print("shape before sector features:", master_df.shape)
print("Columns in master_df:", master_df.columns)


# In[7]:


master_df['date'] = pd.to_datetime(master_df['date'])
master_df = master_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)


# In[8]:


master_df = master_df.sort_values(by=['ticker', 'date'])

master_df['ret_1d_meta'] = (
    master_df.groupby('ticker')['close']
    .pct_change(periods=-H_META)
)


# In[9]:


master_df['up_1d_meta'] = (master_df['ret_1d_meta'] > 0).astype(int)

master_df['has_meta_target'] = master_df['ret_1d_meta'].notna().astype(int)


# In[10]:


tickers = master_df['ticker'].unique()
len(tickers), tickers[:5]


# ## Defining Models

# In[11]:


def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seeds(42)


# Datasets
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNHead(nn.Module):
    # Shared head:
    #   - RNN stack (LSTM/GRU, uni/bi)
    #   - BatchNorm + Dense(32, ReLU) + Dropout
    #   - Output layer (1 unit): linear (regression) or logits (classification)
    def __init__(self, input_size, rnn_type='LSTM', bidirectional=False, problem_type='classification',
                 n_classes=6, ordinal_head='coral', hidden1=128, hidden2=64, num_layers=1,
                 inter_rnn_drop=0.1, dropout=0.3, use_layernorm=False):
        super().__init__()
        self.problem_type = problem_type
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.upper()
        self.n_classes = n_classes
        self.ordinal_head = ordinal_head
        self.num_layers = int(num_layers)
        self.hidden1 = int(hidden1)
        self.hidden2 = int(hidden2)

        if self.num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")

        rnn_cls = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[('GRU' if 'GRU' in self.rnn_type else 'LSTM')]

        self.rnn1 = rnn_cls(
            input_size=input_size, hidden_size=self.hidden1, num_layers=1,
            batch_first=True, dropout=0.0, bidirectional=bidirectional
        )

        self.inter_rnn_drop = nn.Dropout(float(inter_rnn_drop))

        self.rnn2 = None
        if self.num_layers == 2:
            self.rnn2 = rnn_cls(
                input_size=self.hidden1*(2 if bidirectional else 1), hidden_size=self.hidden2, num_layers=1,
                batch_first=True, dropout=0.0, bidirectional=bidirectional
            )
            feat_dim = self.hidden2*(2 if bidirectional else 1)
        else:
            feat_dim = self.hidden1*(2 if bidirectional else 1)

        if use_layernorm:
            self.bn = nn.LayerNorm(feat_dim)
        else:
            self.bn = nn.BatchNorm1d(feat_dim)

        self.fc = nn.Linear(feat_dim, 32)
        self.drop = nn.Dropout(float(dropout))
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.rnn1(x)
        if self.num_layers == 2:
            out = self.inter_rnn_drop(out)   # inter-layer dropout (sequence-wise)
            out, _ = self.rnn2(out)
        # take last timestep: [B, T, H] -> [B, H]
        out = out[:, -1, :]
        out = self.bn(out)
        out = F.relu(self.fc(out))
        out = self.drop(out)
        out = self.out(out)  # shape [B,1]
        return out  # regression: raw; classification: logits
# Early Stopping (PyTorch)
class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        improved = (self.best_loss - val_loss) > self.min_delta
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                # Deep copy state dict
                self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


# In[12]:


def build_model(input_shape, model_type='LSTM', problem_type='regression', hidden1=128, hidden2=64, 
                num_layers=2, inter_rnn_drop=0.1, dropout=0.3):
    seq_len, n_features = input_shape
    model_type = model_type.upper()
    kwargs = dict(
        problem_type=problem_type,
        hidden1=hidden1,
        hidden2=hidden2,
        num_layers=num_layers,
        inter_rnn_drop=inter_rnn_drop,
        dropout=dropout,
    )
    if model_type == 'LSTM':
        return RNNHead(n_features, rnn_type='LSTM', bidirectional=False, **kwargs)
    elif model_type == 'BILSTM':
        return RNNHead(n_features, rnn_type='LSTM', bidirectional=True, **kwargs)
    elif model_type == 'GRU':
        return RNNHead(n_features, rnn_type='GRU', bidirectional=False, **kwargs)
    elif model_type == 'BIGRU':
        return RNNHead(n_features, rnn_type='GRU', bidirectional=True, **kwargs)
    else:
        raise ValueError("Model type must be one of: ['LSTM','BiLSTM','GRU','BiGRU']")


# ## Walk Forward Regression

# In[ ]:


def walk_forward_regression_predictions_per_ticker(
    df_tkr,
    feature_cols,
    target_col='ret_1d_meta',
    model_type='LSTM',
    seq_len=L_REGRESSION,
    w_base=W_BASE,
    refit_interval=REFIT_INTERVAL,
    min_seq=MIN_SEQ,
    hidden1=64,
    hidden2=32,
    num_layers=2,
    inter_rnn_drop=0.4,
    dropout=0.0,
    learning_rate=1e-5,
    weight_decay=4e-4,
    batch_size=32,
    max_epochs=50,
    early_patience=15,
    early_min_delta=0.0,
    huber_delta=1.0,
    device=None
):
    
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_tkr = df_tkr.sort_values('date').reset_index(drop=True)
    df_tkr = df_tkr[~df_tkr[target_col].isna()].copy().reset_index(drop=True)
    if len(df_tkr) < (w_base + refit_interval + seq_len):
        return None, "too_short"

    df_tkr = ffill_only(df_tkr, feature_cols)
    mask_all = df_tkr[feature_cols].notna().all(axis=1)
    if not mask_all.any():
        return None, "no_valid_rows"
    first_good_idx = mask_all.idxmax()
    df_tkr = df_tkr.iloc[first_good_idx:].reset_index(drop=True)
    if len(df_tkr) < (w_base + refit_interval + seq_len):
        return None, "too_short_after_ffill"

    rows = []
    metrics_rows = []
    k = max(w_base - 1, seq_len - 1)

    while True:
        train_end = k
        pred_start = k + 1
        pred_end = min(k + refit_interval, len(df_tkr) - 1)
        if pred_start > pred_end:
            break

        train_df = df_tkr.iloc[:train_end + 1].copy()
        train_df[feature_cols] = train_df[feature_cols].ffill()

        scaler = fit_scaler_on_train(train_df, feature_cols)
        train_df_s = transform_df(train_df, scaler, feature_cols)

        seq_X_train, seq_y_train, _ = build_sequences_meta(train_df_s, feature_cols, target_col, seq_len)
        if len(seq_X_train) < min_seq:
            return None, "too_few_sequences"

        val_cut = max(int(len(seq_X_train) * 0.2), 1)
        n_train = len(seq_X_train) - val_cut
        X_tr, y_tr = seq_X_train[:n_train], seq_y_train[:n_train]
        X_val, y_val = seq_X_train[n_train:], seq_y_train[n_train:]
        if len(X_val) == 0:
            X_val, y_val = X_tr, y_tr

        train_ds = SequenceDataset(X_tr, y_tr)
        val_ds   = SequenceDataset(X_val, y_val)

        train_bs = min(int(batch_size), len(train_ds))
        if train_bs < 2:
            return None, "train_batch_too_small"
        if len(train_ds) % train_bs == 1 and train_bs > 2:
            train_bs -= 1
        val_bs = min(int(batch_size), len(val_ds))

        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False, drop_last=True)

        model = build_model((seq_len, X_tr.shape[-1]), model_type=model_type, problem_type='regression', num_layers=num_layers, hidden1=hidden1, hidden2=hidden2, inter_rnn_drop=inter_rnn_drop, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        loss_fn = nn.HuberLoss(delta=float(huber_delta))
        early = EarlyStopper(patience=int(early_patience), min_delta=float(early_min_delta), restore_best=True)

        for epoch in range(int(max_epochs)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device).view(-1, 1)
                optimizer.zero_grad(set_to_none=True)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vtot, n = 0.0, 0
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).view(-1, 1)
                    preds = model(xb)
                    loss = loss_fn(preds, yb)
                    vtot += loss.item() * xb.size(0)
                    n += xb.size(0)
                val_loss = vtot / max(n, 1)

            if early.step(val_loss, model):
                break

        early.restore(model)

        block_end = pred_end
        block_df = df_tkr.iloc[:block_end + 1].copy()
        block_df[feature_cols] = block_df[feature_cols].ffill()
        block_df_s = transform_df(block_df, scaler, feature_cols)

        seq_X_all, seq_y_all, idx_all = build_sequences_meta(block_df_s, feature_cols, target_col, seq_len)
        if len(seq_X_all) == 0:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            break

        end_dates = block_df_s.iloc[idx_all]['date'].values
        block_dates = df_tkr.iloc[pred_start:pred_end+1]['date'].values
        block_mask = np.isin(end_dates, block_dates)

        if block_mask.any():
            X_block = torch.tensor(seq_X_all[block_mask], dtype=torch.float32).to(device)
            y_block = seq_y_all[block_mask]
            model.eval()
            with torch.no_grad():
                preds = model(X_block).view(-1).cpu().numpy()

            mae = mean_absolute_error(y_block, preds)
            mse = mean_squared_error(y_block, preds)
            y_true_dir = (y_block > 0).astype(int)
            y_pred_dir = (preds > 0).astype(int)
            precision = precision_score(y_true_dir, y_pred_dir, zero_division=0)
            recall = recall_score(y_true_dir, y_pred_dir, zero_division=0)
            f1 = f1_score(y_true_dir, y_pred_dir, zero_division=0)
            acc = (y_true_dir == y_pred_dir).mean()
            mcc = matthews_corrcoef(y_true_dir, y_pred_dir)

            metrics_rows.append({
                'ticker': df_tkr.loc[0, 'ticker'],
                'refit_date': df_tkr.iloc[train_end]['date'],
                'block_end': df_tkr.iloc[pred_end]['date'],
                'mae': mae,
                'mse': mse,
                'dir_acc': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'mcc': mcc,
                'n_preds': int(block_mask.sum())
            })

            for d, p in zip(end_dates[block_mask], preds):
                rows.append({'date': d, 'ticker': df_tkr.loc[0, 'ticker'], 'regression_pred_ret_1d': float(p)})

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        k += refit_interval
        if k >= len(df_tkr) - 2:
            break

    pred_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metrics_rows)
    return pred_df, metrics_df


# ## Walk Forward Classification

# In[ ]:


def walk_forward_classification_predictions_per_ticker(
    df_tkr,
    feature_cols,
    target_col='up_1d_meta',
    model_type='GRU',
    seq_len=L_CLASSIFICATION,
    w_base=W_BASE_CLASSIFICATION,
    refit_interval=REFIT_INTERVAL_CLASSIFICATION,
    min_seq=MIN_SEQ_CLASSIFICATION,
    hidden1=64,
    hidden2=32,
    num_layers=1,
    inter_rnn_drop=0.0,
    dropout=0.4,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=20,
    early_patience=5,
    early_min_delta=0.0,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_tkr = df_tkr.sort_values('date').reset_index(drop=True)
    df_tkr = df_tkr[~df_tkr[target_col].isna()].copy().reset_index(drop=True)
    if len(df_tkr) < (w_base + refit_interval + seq_len):
        return None, "too_short"

    df_tkr = ffill_only(df_tkr, feature_cols)
    mask_all = df_tkr[feature_cols].notna().all(axis=1)
    if not mask_all.any():
        return None, "no_valid_rows"
    first_good_idx = mask_all.idxmax()
    df_tkr = df_tkr.iloc[first_good_idx:].reset_index(drop=True)
    if len(df_tkr) < (w_base + refit_interval + seq_len):
        return None, "too_short_after_ffill"

    rows = []
    metrics_rows = []
    k = max(w_base - 1, seq_len - 1)

    while True:
        train_end = k
        pred_start = k + 1
        pred_end = min(k + refit_interval, len(df_tkr) - 1)
        if pred_start > pred_end:
            break

        train_df = df_tkr.iloc[:train_end + 1].copy()
        train_df[feature_cols] = train_df[feature_cols].ffill()

        scaler = fit_scaler_on_train(train_df, feature_cols)
        train_df_s = transform_df(train_df, scaler, feature_cols)

        seq_X_train, seq_y_train, _ = build_sequences_meta(train_df_s, feature_cols, target_col, seq_len)
        if len(seq_X_train) < min_seq:
            return None, "too_few_sequences"

        val_cut = max(int(len(seq_X_train) * 0.2), 1)
        n_train = len(seq_X_train) - val_cut
        X_tr, y_tr = seq_X_train[:n_train], seq_y_train[:n_train]
        X_val, y_val = seq_X_train[n_train:], seq_y_train[n_train:]
        if len(X_val) == 0:
            X_val, y_val = X_tr, y_tr

        pos_rate = y_tr.mean()
        if pos_rate > 0 and pos_rate < 1:
            pos_weight = (1.0 - pos_rate) / pos_rate
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        else:
            pos_weight_tensor = None

        train_ds = SequenceDataset(X_tr, y_tr)
        val_ds   = SequenceDataset(X_val, y_val)

        train_bs = min(int(batch_size), len(train_ds))
        if train_bs < 2:
            return None, "train_batch_too_small"
        if len(train_ds) % train_bs == 1 and train_bs > 2:
            train_bs -= 1
        val_bs = min(int(batch_size), len(val_ds))

        train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False, drop_last=True)

        model = build_model((seq_len, X_tr.shape[-1]), model_type=model_type, problem_type='classification', num_layers=num_layers, hidden1=hidden1, hidden2=hidden2, inter_rnn_drop=inter_rnn_drop, dropout=dropout).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        early = EarlyStopper(patience=int(early_patience), min_delta=float(early_min_delta), restore_best=True)

        for epoch in range(int(max_epochs)):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device).view(-1, 1)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                vtot, n = 0.0, 0
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device).view(-1, 1)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    vtot += loss.item() * xb.size(0)
                    n += xb.size(0)
                val_loss = vtot / max(n, 1)

            if early.step(val_loss, model):
                break

        early.restore(model)

        block_end = pred_end
        block_df = df_tkr.iloc[:block_end + 1].copy()
        block_df[feature_cols] = block_df[feature_cols].ffill()
        block_df_s = transform_df(block_df, scaler, feature_cols)

        seq_X_all, seq_y_all, idx_all = build_sequences_meta(block_df_s, feature_cols, target_col, seq_len)
        if len(seq_X_all) == 0:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            break

        end_dates = block_df_s.iloc[idx_all]['date'].values
        block_dates = df_tkr.iloc[pred_start:pred_end+1]['date'].values
        block_mask = np.isin(end_dates, block_dates)

        if block_mask.any():
            X_block = torch.tensor(seq_X_all[block_mask], dtype=torch.float32).to(device)
            y_block = seq_y_all[block_mask]
            model.eval()
            with torch.no_grad():
                logits_block = model(X_block).view(-1).cpu().numpy()
            probs_block = 1.0 / (1.0 + np.exp(-logits_block))

            eps = 1e-8
            brier = (probs_block - y_block) ** 2
            logloss = - (y_block * np.log(probs_block + eps) + (1 - y_block) * np.log(1 - probs_block + eps))
            correct = (probs_block >= 0.5).astype(int) == y_block

            metrics_rows.append({
                'ticker': df_tkr.loc[0, 'ticker'],
                'refit_date': df_tkr.iloc[train_end]['date'],
                'block_end': df_tkr.iloc[pred_end]['date'],
                'brier': float(brier.mean()),
                'logloss': float(logloss.mean()),
                'acc': float(correct.mean()),
                'n_preds': int(block_mask.sum())
            })

            for d, p in zip(end_dates[block_mask], probs_block):
                rows.append({'date': d, 'ticker': df_tkr.loc[0, 'ticker'], 'prob_up_1d': float(p)})

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        k += refit_interval
        if k >= len(df_tkr) - 2:
            break

    pred_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metrics_rows)
    return pred_df, metrics_df


# ## Pipeline Regression

# In[ ]:


# Select best configs (highest mean MCC) per model for H=1
MODELS = ['LSTM', 'BiLSTM', 'GRU', 'BiGRU']
classification_best = select_best_configs('classification', models=MODELS)

# Regression tuning results might not exist; fallback to classification configs
regression_base_dir = RESULTS_ROOT / 'benchmarking' / 'regression'
if not regression_base_dir.exists():
    regression_base_dir = Path('results/benchmarking/regression')
if regression_base_dir.exists():
    regression_best = select_best_configs('regression', models=MODELS)
else:
    print("[WARN] No regression tuning results found under results/benchmarking/regression. Using classification-best configs for regression meta models.")
    regression_best = classification_best

# Print selected configs (ordered)
print("\n=== SELECTED CONFIGS (H=1, sorted by MCC) ===")
for model in MODELS:
    cfg = classification_best[model]
    params = cfg['params']
    print(f"[CLS] {model:<6} | feature_group={cfg['feature_group']:<6} | feature_set={cfg['feature_set']:<18} | mcc_mean={cfg['mcc_mean']:.4f}")
    print(f"      params: seq={params.get('params_sequence_length')} hidden1={params.get('params_hidden1')} hidden2={params.get('params_hidden2')} layers={params.get('params_num_layers')} drop={params.get('params_dropout')} inter_drop={params.get('params_inter_rnn_drop')} lr={params.get('params_learning_rate')} wd={params.get('params_weight_decay')} bs={params.get('params_batch_size')}")

if regression_best is classification_best:
    print("[REG] Using classification-selected configs (no regression tuning results found).")
else:
    for model in MODELS:
        cfg = regression_best[model]
        params = cfg['params']
        print(f"[REG] {model:<6} | feature_group={cfg['feature_group']:<6} | feature_set={cfg['feature_set']:<18} | mcc_mean={cfg['mcc_mean']:.4f}")
        print(f"      params: seq={params.get('params_sequence_length')} hidden1={params.get('params_hidden1')} hidden2={params.get('params_hidden2')} layers={params.get('params_num_layers')} drop={params.get('params_dropout')} inter_drop={params.get('params_inter_rnn_drop')} lr={params.get('params_learning_rate')} wd={params.get('params_weight_decay')} bs={params.get('params_batch_size')}")

# Run walk-forward regression for each model
print(f"\n=== REGRESSION META: refit interval {REFIT_INTERVAL}, warm-up {W_BASE} ===")

all_reg_preds = []
all_reg_metrics = []
skip_counts_reg = {}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

for model in MODELS:
    cfg = regression_best[model]
    params = cfg['params']
    feature_set_name = _resolve_feature_set_name(str(cfg['feature_set']))
    feature_cols = feature_sets[feature_set_name]
    _validate_feature_cols(feature_cols, master_df.columns)

    seq_len = _get_param(params, 'params_sequence_length', L_REGRESSION, int)
    hidden1 = _get_param(params, 'params_hidden1', 64, int)
    hidden2 = _get_param(params, 'params_hidden2', 32, int)
    num_layers = _get_param(params, 'params_num_layers', 2, int)
    inter_rnn_drop = _get_param(params, 'params_inter_rnn_drop', 0.4, float)
    dropout = _get_param(params, 'params_dropout', 0.0, float)
    learning_rate = _get_param(params, 'params_learning_rate', 1e-5, float)
    weight_decay = _get_param(params, 'params_weight_decay', 4e-4, float)
    batch_size = 32
    max_epochs = _get_param(params, 'params_max_epochs', 50, int)
    early_patience = _get_param(params, 'params_early_stopping_patience', 15, int)
    early_min_delta = _get_param(params, 'params_early_stopping_min_delta', 0.0, float)
    huber_delta = _get_param(params, 'params_huber_delta', 1.0, float)

    print(f"\n[REG-MODEL] {model} | feature_set={feature_set_name} | seq={seq_len} | h1={hidden1} h2={hidden2} | layers={num_layers} | lr={learning_rate} wd={weight_decay} | bs={batch_size}")

    for i, tkr in enumerate(tickers, 1):
        print(f"  [REG] {model} ticker {i}/{len(tickers)}: {tkr}")
        df_tkr = master_df[master_df['ticker'] == tkr].copy()
        pred_df, metrics_df = walk_forward_regression_predictions_per_ticker(
            df_tkr,
            feature_cols=feature_cols,
            target_col='ret_1d_meta',
            model_type=model,
            seq_len=seq_len,
            w_base=W_BASE,
            refit_interval=REFIT_INTERVAL,
            min_seq=MIN_SEQ,
            hidden1=hidden1,
            hidden2=hidden2,
            num_layers=num_layers,
            inter_rnn_drop=inter_rnn_drop,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_patience=early_patience,
            early_min_delta=early_min_delta,
            huber_delta=huber_delta,
            device=device
        )
        if pred_df is None or pred_df.empty:
            reason = metrics_df if isinstance(metrics_df, str) else 'no_predictions'
            skip_counts_reg[(model, reason)] = skip_counts_reg.get((model, reason), 0) + 1
            continue

        pred_df = pred_df.rename(columns={'regression_pred_ret_1d': f'reg_pred_ret_1d_{model}'})
        all_reg_preds.append(pred_df)
        if metrics_df is not None and not metrics_df.empty:
            metrics_df = metrics_df.copy()
            metrics_df['model'] = model
            all_reg_metrics.append(metrics_df)

# Aggregate regression metrics
if all_reg_metrics:
    metrics_all = pd.concat(all_reg_metrics, ignore_index=True)
    agg = metrics_all[['mae','mse','dir_acc','f1','precision','recall','mcc']].mean().to_dict()
    total_preds = metrics_all['n_preds'].sum()
    print("\n=== REGRESSION META SUMMARY ===")
    print(f"MAE: {agg['mae']:.6f} | MSE: {agg['mse']:.6f} | Dir Acc: {agg['dir_acc']:.4f} | F1: {agg['f1']:.4f} | Precision: {agg['precision']:.4f} | Recall: {agg['recall']:.4f} | MCC: {agg['mcc']:.4f}")
    print(f"Refits: {len(metrics_all)} | Total preds: {total_preds}")

print("Regression skip summary:", skip_counts_reg)


# ### Regression Meta Features

# In[ ]:


# Merge regression predictions and compute error/reliability features

# drop any existing regression meta columns
reg_pred_cols = [c for c in master_df.columns if c.startswith('reg_pred_ret_1d_')]
reg_err_cols = [c for c in master_df.columns if c.startswith('reg_')]
cols_to_drop = sorted(set(reg_pred_cols + reg_err_cols))
if cols_to_drop:
    master_df = master_df.drop(columns=cols_to_drop)

# merge new regression predictions
if all_reg_preds:
    reg_all = pd.concat(all_reg_preds, ignore_index=True)
    pred_cols = [c for c in reg_all.columns if c not in ['date','ticker']]
    reg_all = reg_all.sort_values(['date','ticker'])
    reg_all = reg_all.groupby(['date','ticker'], as_index=False)[pred_cols].last()
    master_df = master_df.merge(reg_all, on=['date', 'ticker'], how='left')

print('master_df with regression meta:', master_df.shape)

# compute regression prediction errors (diagnostics) per model
master_df = master_df.sort_values(['ticker', 'date']).reset_index(drop=True)
for model in MODELS:
    pred_col = f'reg_pred_ret_1d_{model}'
    if pred_col not in master_df.columns:
        continue
    mask = master_df[pred_col].notna() & master_df['ret_1d_meta'].notna()
    err_col = f'reg_err_ret_1d_{model}'
    abs_col = f'reg_abs_err_ret_1d_{model}'
    sq_col  = f'reg_sq_err_ret_1d_{model}'
    dir_col = f'reg_dir_correct_1d_{model}'

    master_df.loc[mask, err_col] = master_df.loc[mask, pred_col] - master_df.loc[mask, 'ret_1d_meta']
    master_df.loc[mask, abs_col] = master_df.loc[mask, err_col].abs()
    master_df.loc[mask, sq_col]  = master_df.loc[mask, err_col] ** 2
    master_df.loc[mask, dir_col] = (
        (master_df.loc[mask, pred_col] > 0).astype(int) == master_df.loc[mask, 'up_1d_meta']
    ).astype(int)

    # leak-safe reliability features (shifted/rolling)
    master_df[f'reg_abs_err_lag1_{model}'] = master_df.groupby('ticker')[abs_col].shift(1)
    master_df[f'reg_mae_20_{model}'] = (
        master_df.groupby('ticker')[abs_col]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    master_df[f'reg_rmse_20_{model}'] = (
        master_df.groupby('ticker')[sq_col]
        .transform(lambda s: np.sqrt(s.shift(1).rolling(20, min_periods=5).mean()))
    )
    master_df[f'reg_dir_acc_20_{model}'] = (
        master_df.groupby('ticker')[dir_col]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )

print("Regression reliability features computed. Sample:")
cols_preview = ['date','ticker'] + [c for c in master_df.columns if c.startswith('reg_pred_ret_1d_')][:2]
print(master_df[cols_preview].head())


# ## Pipeline Classification

# In[ ]:


print(f"\n=== CLASSIFICATION META: refit interval {REFIT_INTERVAL_CLASSIFICATION}, warm-up {W_BASE_CLASSIFICATION} ===")

all_class_preds = []
all_class_metrics = []
skip_counts_class = {}

for model in MODELS:
    cfg = classification_best[model]
    params = cfg['params']
    feature_set_name = _resolve_feature_set_name(str(cfg['feature_set']))
    feature_cols = feature_sets[feature_set_name]
    _validate_feature_cols(feature_cols, master_df.columns)

    seq_len = _get_param(params, 'params_sequence_length', L_CLASSIFICATION, int)
    hidden1 = _get_param(params, 'params_hidden1', 64, int)
    hidden2 = _get_param(params, 'params_hidden2', 32, int)
    num_layers = _get_param(params, 'params_num_layers', 1, int)
    inter_rnn_drop = _get_param(params, 'params_inter_rnn_drop', 0.0, float)
    dropout = _get_param(params, 'params_dropout', 0.4, float)
    learning_rate = _get_param(params, 'params_learning_rate', 1e-3, float)
    weight_decay = _get_param(params, 'params_weight_decay', 1e-4, float)
    batch_size = 32
    max_epochs = _get_param(params, 'params_max_epochs', 20, int)
    early_patience = _get_param(params, 'params_early_stopping_patience', 5, int)
    early_min_delta = _get_param(params, 'params_early_stopping_min_delta', 0.0, float)

    print(f"\n[CLS-MODEL] {model} | feature_set={feature_set_name} | seq={seq_len} | h1={hidden1} h2={hidden2} | layers={num_layers} | lr={learning_rate} wd={weight_decay} | bs={batch_size}")

    for i, tkr in enumerate(tickers, 1):
        print(f"  [CLS] {model} ticker {i}/{len(tickers)}: {tkr}")
        df_tkr = master_df[master_df['ticker'] == tkr].copy()
        pred_df, metrics_df = walk_forward_classification_predictions_per_ticker(
            df_tkr,
            feature_cols=feature_cols,
            target_col='up_1d_meta',
            model_type=model,
            seq_len=seq_len,
            w_base=W_BASE_CLASSIFICATION,
            refit_interval=REFIT_INTERVAL_CLASSIFICATION,
            min_seq=MIN_SEQ_CLASSIFICATION,
            hidden1=hidden1,
            hidden2=hidden2,
            num_layers=num_layers,
            inter_rnn_drop=inter_rnn_drop,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_patience=early_patience,
            early_min_delta=early_min_delta,
            device=device
        )
        if pred_df is None or pred_df.empty:
            reason = metrics_df if isinstance(metrics_df, str) else 'no_predictions'
            skip_counts_class[(model, reason)] = skip_counts_class.get((model, reason), 0) + 1
            continue

        pred_df = pred_df.rename(columns={'prob_up_1d': f'cls_prob_up_1d_{model}'})
        all_class_preds.append(pred_df)
        if metrics_df is not None and not metrics_df.empty:
            metrics_df = metrics_df.copy()
            metrics_df['model'] = model
            all_class_metrics.append(metrics_df)

# Aggregate classification metrics
if all_class_metrics:
    metrics_all = pd.concat(all_class_metrics, ignore_index=True)
    agg = metrics_all[['brier','logloss','acc']].mean().to_dict()
    total_preds = metrics_all['n_preds'].sum()
    print("\n=== CLASSIFICATION META SUMMARY ===")
    print(f"Brier: {agg['brier']:.6f} | Logloss: {agg['logloss']:.6f} | Acc: {agg['acc']:.4f}")
    print(f"Refits: {len(metrics_all)} | Total preds: {total_preds}")

print("Classification skip summary:", skip_counts_class)


# ### Classification Meta Features

# In[ ]:


# Merge classification predictions and compute error/reliability features

# drop any existing classification meta columns
cls_pred_cols = [c for c in master_df.columns if c.startswith('cls_prob_up_1d_')]
cls_err_cols = [c for c in master_df.columns if c.startswith('cls_')]
cols_to_drop = sorted(set(cls_pred_cols + cls_err_cols))
if cols_to_drop:
    master_df = master_df.drop(columns=cols_to_drop)

# merge new classification predictions
if all_class_preds:
    cls_all = pd.concat(all_class_preds, ignore_index=True)
    pred_cols = [c for c in cls_all.columns if c not in ['date','ticker']]
    cls_all = cls_all.sort_values(['date','ticker'])
    cls_all = cls_all.groupby(['date','ticker'], as_index=False)[pred_cols].last()
    master_df = master_df.merge(cls_all, on=['date', 'ticker'], how='left')

print('master_df with Classification meta:', master_df.shape)

# diagnostics and leak-safe reliability features for Classification per model
master_df = master_df.sort_values(['ticker','date']).reset_index(drop=True)
for model in MODELS:
    prob_col = f'cls_prob_up_1d_{model}'
    if prob_col not in master_df.columns:
        continue
    mask_cls = master_df[prob_col].notna() & master_df['up_1d_meta'].notna()
    eps = 1e-8
    brier_col = f'cls_brier_{model}'
    logloss_col = f'cls_logloss_{model}'
    correct_col = f'cls_correct_{model}'

    master_df.loc[mask_cls, brier_col] = (master_df.loc[mask_cls, prob_col] - master_df.loc[mask_cls, 'up_1d_meta']) ** 2
    master_df.loc[mask_cls, logloss_col] = -(
        master_df.loc[mask_cls, 'up_1d_meta'] * np.log(master_df.loc[mask_cls, prob_col] + eps)
        + (1 - master_df.loc[mask_cls, 'up_1d_meta']) * np.log(1 - master_df.loc[mask_cls, prob_col] + eps)
    )
    master_df.loc[mask_cls, correct_col] = (
        (master_df.loc[mask_cls, prob_col] >= 0.5).astype(int) == master_df.loc[mask_cls, 'up_1d_meta']
    ).astype(int)

    master_df[f'cls_brier_20_{model}'] = (
        master_df.groupby('ticker')[brier_col]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    master_df[f'cls_logloss_20_{model}'] = (
        master_df.groupby('ticker')[logloss_col]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )
    master_df[f'cls_acc_20_{model}'] = (
        master_df.groupby('ticker')[correct_col]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    )

print("Classification reliability features computed. Sample:")
cols_preview = ['date','ticker'] + [c for c in master_df.columns if c.startswith('cls_prob_up_1d_')][:2]
print(master_df[cols_preview].head())

# save combined parquet with Regression and Classification signals
# drop leak-prone diagnostics from output (keep lag/rolling reliability features)
leaky_cols = []
for model in MODELS:
    leaky_cols += [
        f'reg_err_ret_1d_{model}',
        f'reg_abs_err_ret_1d_{model}',
        f'reg_sq_err_ret_1d_{model}',
        f'reg_dir_correct_1d_{model}',
        f'cls_brier_{model}',
        f'cls_logloss_{model}',
        f'cls_correct_{model}',
    ]

meta_df = master_df.drop(columns=[c for c in leaky_cols if c in master_df.columns], errors='ignore')

out_path = META_OUT_PATH
meta_df.to_parquet(out_path, index=False)
print("Saved:", out_path)
print("Dropped leaky cols:", [c for c in leaky_cols if c in master_df.columns])


# In[ ]:


# Quick visual sanity check for one model (best MCC)

try:
    plot_model = max(regression_best, key=lambda m: regression_best[m]['mcc_mean'])
except Exception:
    plot_model = 'LSTM'

pred_col = f'reg_pred_ret_1d_{plot_model}'
abs_err_col = f'reg_abs_err_ret_1d_{plot_model}'

random.seed(42)
sector_ticker_map = {}
for sector in master_df['sector'].dropna().unique():
    tickers_in_sector = master_df.loc[master_df['sector'] == sector, 'ticker'].dropna().unique()
    if len(tickers_in_sector):
        sector_ticker_map[sector] = random.choice(tickers_in_sector)

for sector, ticker in sector_ticker_map.items():
    stock_data = master_df[master_df['ticker'] == ticker].dropna(subset=[pred_col, 'ret_1d_meta'])
    if stock_data.empty:
        print(f"No valid data for sector {sector}, ticker {ticker}. Skipping.")
        continue
    stock_data = stock_data.sort_values('date')
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['date'], stock_data['ret_1d_meta'], label='Actual Return', color='blue')
    plt.plot(stock_data['date'], stock_data[pred_col], label=f'Regression Pred Return ({plot_model})', color='orange')
    if abs_err_col in stock_data.columns:
        plt.fill_between(
            stock_data['date'],
            stock_data[pred_col] - stock_data[abs_err_col],
            stock_data[pred_col] + stock_data[abs_err_col],
            color='orange', alpha=0.2, label='|Error| Band'
        )
    plt.title(f"{sector} - {ticker}: Actual vs Predicted Return ({plot_model})")
    plt.xlabel("Date")
    plt.ylabel("1-day return")
    plt.legend()
    plt.grid()
    plt.show()

