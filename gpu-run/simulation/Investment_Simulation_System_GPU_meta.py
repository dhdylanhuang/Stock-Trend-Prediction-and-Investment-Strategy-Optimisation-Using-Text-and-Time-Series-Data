#!/usr/bin/env python
# coding: utf-8

# # Investment Simulation System

# ### Imports

# In[1]:


import os
import site

import os
import shutil
import gc
import warnings
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from datetime import timedelta

warnings.filterwarnings('ignore')


# In[2]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)


# In[3]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext


# In[ ]:


from pathlib import Path

PERSIST_ROOT = Path(os.environ.get('PERSIST_ROOT', '/mnt/primary'))
if not PERSIST_ROOT.exists():
    raise RuntimeError(f'Persistent storage not found at {PERSIST_ROOT}. Check mounts (df -h /mnt/primary).')

RUN_ROOT = Path(os.environ.get('RUN_ROOT', PERSIST_ROOT / 'simulation'))
if not str(RUN_ROOT).startswith(str(PERSIST_ROOT)):
    print(f'WARNING: RUN_ROOT={RUN_ROOT} is not on persistent storage; forcing to {PERSIST_ROOT}/simulation')
    RUN_ROOT = PERSIST_ROOT / 'simulation'
RUN_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_PATH = Path(os.environ.get('MASTER_PATH', RUN_ROOT / 'master_df_60rf.parquet'))
EARLY_EXIT_PATH = Path(os.environ.get('EARLY_EXIT_PATH', PERSIST_ROOT / 'early-exit' / 'h1_exit_df.parquet'))
MODEL_SAVE_PATH = Path(os.environ.get('MODEL_SAVE_PATH', RUN_ROOT / 'trained_models_meta'))
OUTPUT_ROOT = Path(os.environ.get('OUTPUT_ROOT', RUN_ROOT / 'results'))
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print('MASTER_PATH:', MASTER_PATH)
print('EARLY_EXIT_PATH:', EARLY_EXIT_PATH)
print('MODEL_SAVE_PATH:', MODEL_SAVE_PATH)
print('OUTPUT_ROOT:', OUTPUT_ROOT)

FEATURE_SET_ALIAS = {
    'sentinment': 'sentiment',
}


# ### Configurations

# In[ ]:


MODEL_SAVE_PATH = MODEL_SAVE_PATH if 'MODEL_SAVE_PATH' in globals() else Path('trained_models_meta/')
MIN_SEQUENCE_LENGTH = 12  # Minimum sequence length for any company
MAX_SEQUENCE_LENGTH = 12  # Maximum sequence length to cap computational cost
INITIAL_TRAINING_DAYS = 853  # Number of days to use for initial training only
KELLY_FRACTION = 0.20
SECTOR_CONFIDENCE_THRESHOLD = 0.30
RETRAIN_INTERVAL = 200
MAX_DAY_GAP = 5  # Maximum allowed gap in trading days (to account for weekends/holidays)

ALLOCATION_MODE = "KELLY_ONLY"  # {"KELLY_ONLY", "HYBRID_KELLY_MPT", "MPT_ONLY"}
MPT_WINDOW_DAYS = 60
MPT_LAMBDA = 1.0
MPT_RIDGE_EPS = 1e-4
MPT_MAX_WEIGHT = 0.30
MPT_LONG_ONLY = True
MPT_USE_EQUAL_FALLBACK = True
MPT_ONLY_BUDGET_FRACTION = 0.3

VERBOSE = True


# In[ ]:


MAX_TOTAL_UTILIZATION      = 0.7   # never deploy more than 70% of capital at once
MAX_NEW_DAILY_UTILIZATION  = 0.30   # new trades today ≤ 30% of capital
MAX_TICKER_UTILIZATION     = 0.50   # per ticker cap (sum of all overlapping trades)
MAX_SECTOR_UTILIZATION     = 0.50   # per sector cap (sum across its tickers)
MAX_POSITION_SIZE          = 0.1   # single-trade cap

MAX_POSITIONS_PER_TICKER   = 3      # ladder depth per ticker
MIN_TRADE_DOLLARS          = 100.0  # skip dust trades

KELLY_TRADE_CAP            = 0.50   # cap raw Kelly per trade (before other caps)
DRAWDOWN_THROTTLE_LEVELS   = [(0.08, 0.60), (0.15, 0.35)]  # (peak_dd, utilization_multiplier)


# In[7]:


import random


def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seeds(42)


# In[8]:


# GPU-focused execution knobs
USE_MIXED_PRECISION = torch.cuda.is_available()
MATMUL_PRECISION = os.environ.get('MATMUL_PRECISION', 'high')
if hasattr(torch, 'set_float32_matmul_precision'):
    try:
        torch.set_float32_matmul_precision(MATMUL_PRECISION)
    except Exception:
        pass

NUM_WORKERS_TRAIN = 2
NUM_WORKERS_EVAL  = 2
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = (NUM_WORKERS_TRAIN > 0)


# #### Define Horizon Target
# For a horizon H, compute both the direction and H-day return off the same base day t

# In[9]:


def add_horizon_targets(df: pd.DataFrame, H: int, price_col='close') -> pd.DataFrame:
    df = df.sort_values(['ticker','date']).copy()
    df[f'ret_{H}d']    = df.groupby('ticker')[price_col].shift(-H) / df[price_col] - 1.0
    df[f'target_{H}d'] = (df[f'ret_{H}d'] > 0).astype(int)
    return df


# #### Identify Contiguous Periods
# Groups rows for a ticker into blocks where data gaps never exceed MAX_DAY_GAP

# In[10]:


def identify_contiguous_periods(df: pd.DataFrame, max_gap_days: int = MAX_DAY_GAP) -> list:

    if df.empty:
        return []

    df = df.sort_values('date').reset_index(drop=True)
    dates = pd.to_datetime(df['date'])

    contiguous_periods = []
    start_idx = 0

    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        if gap > max_gap_days:
            # End current period and start new one
            contiguous_periods.append((start_idx, i-1))
            start_idx = i


    contiguous_periods.append((start_idx, len(dates)-1))


    contiguous_periods = [(s, e) for s, e in contiguous_periods if e - s >= MIN_SEQUENCE_LENGTH]

    return contiguous_periods


# #### Calculate Dynamic Sequence Length 
# Pick a sequence length tailored to the amount of history available for a ticker.
# Avoids hard-coding a single window size for short vs long histories.

# In[11]:


def calculate_dynamic_sequence_length(company_data_df: pd.DataFrame,
                                     min_length: int = MIN_SEQUENCE_LENGTH,
                                     max_length: int = MAX_SEQUENCE_LENGTH,
                                     target_fraction: float = 0.15) -> int:

    total_days = len(company_data_df)


    dynamic_length = int(total_days * target_fraction)
    dynamic_length = max(min_length, min(dynamic_length, max_length))

    return dynamic_length


# #### Create Contiguous Sequences
# Turn each contiguous period into overlapping (sequence_length) windows (X) and next-step labels (y).
# Produces the actual training/validation tensors for the model.

# In[12]:


def create_contiguous_sequences(data: np.ndarray, targets: np.ndarray,
                               contiguous_periods: list, sequence_length: int, H: int):

    X, y = [], []

    for start_idx, end_idx in contiguous_periods:
        usable_end = end_idx - H
        period_length = usable_end - start_idx + 1
        if period_length < sequence_length + 1:
            continue

        for i in range(start_idx + sequence_length -1 , usable_end + 1):
            X.append(data[i-sequence_length + 1 : i +1])
            y.append(targets[i + H])

    return np.array(X) if X else np.array([]), np.array(y) if y else np.array([])


# #### Create Target Variable
# Build the binary classification target per row.

# In[13]:


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:

    print("Creating target variable...")
    df = df.sort_values(by=['ticker', 'date']).copy()
    df['next_day_close'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['next_day_close'] > df['close']).astype(int)
    df.dropna(subset=['next_day_close'], inplace=True)
    df['target'] = df['target'].astype(int)
    print("Target variable created.")
    return df


# #### Track Current Exposures from Open Trades
# Settle exits first each day, then compute exposure books before opening new ones.

# In[14]:


from collections import defaultdict

def compute_exposures(open_positions, sectors_by_ticker):
    expo_ticker = defaultdict(float)
    expo_sector = defaultdict(float)
    total = 0.0
    for pos in open_positions:
        amt = pos['invest']
        tkr = pos['ticker']
        sec = sectors_by_ticker.get(tkr, 'UNKNOWN')
        expo_ticker[tkr] += amt
        expo_sector[sec] += amt
        total += amt
    return total, dict(expo_ticker), dict(expo_sector)


# #### Throttle Book When in Drawdown 
# Apply multiplicative haircut to utilisation caps based on current peak to trough drawdown 

# In[15]:


def utilization_throttle(equity_curve):
    if not equity_curve:
        return 1.0
    peak = max(equity_curve)
    curr = equity_curve[-1]
    dd = 0.0 if peak == 0 else (peak - curr)/peak
    mult = 1.0
    for level, m in DRAWDOWN_THROTTLE_LEVELS:
        if dd >= level:
            mult = min(mult, m)
    return mult


# #### Cap Each Proposed Trade
# When sizing a new trade, clamp it by all remaining budgets

# In[16]:


def clamp_by_caps(proposed_amt, capital, 
                  total_open, new_today_open, expo_ticker, expo_sector,
                  ticker, sector,
                  util_mult=1.0):
    # remaining dollar budgets
    rem_total  = MAX_TOTAL_UTILIZATION*util_mult*capital - total_open
    rem_new    = MAX_NEW_DAILY_UTILIZATION*util_mult*capital - new_today_open
    rem_ticker = MAX_TICKER_UTILIZATION*util_mult*capital - expo_ticker.get(ticker, 0.0)
    rem_sector = MAX_SECTOR_UTILIZATION*util_mult*capital - expo_sector.get(sector, 0.0)
    per_trade  = MAX_POSITION_SIZE*util_mult*capital

    # clamp
    max_affordable = max(0.0, min(proposed_amt, rem_total, rem_new, rem_ticker, rem_sector, per_trade))
    return max_affordable


# #### Calculate Historical Payouts
# Estimate average upside “b” per winning trade for Kelly sizing. 
# 
# Only looks at wins; losses are implicit in Kelly’s (1-p)/b term.
# 

# In[17]:


def calculate_historical_payouts(df, ret_col: str = None):
    """
    Returns a dict per ticker with:
      - 'b': avg_win / avg_loss_abs  (Kelly 'odds' ratio)
      - 'avg_win': mean positive return for the chosen horizon
      - 'avg_loss_abs': mean absolute negative return for the chosen horizon
    Uses the same horizon return column as the multiclass model (ret_{H}d).
    """
    if ret_col is None:
        ret_cols = [c for c in df.columns if c.startswith('ret_') and c.endswith('d')]
        ret_col = ret_cols[0] if ret_cols else None
    if ret_col is None or ret_col not in df.columns:
        raise ValueError("Return column for payout computation not found.")

    out = {}
    for t, g in df.groupby('ticker'):
        r = g[ret_col]
        wins = r[r > 0]
        losses = r[r < 0]
        avg_win = wins.mean() if not wins.empty else np.nan
        avg_loss_abs = (-losses).mean() if not losses.empty else np.nan

        if np.isfinite(avg_win) and np.isfinite(avg_loss_abs) and avg_loss_abs > 0:
            b = avg_win / avg_loss_abs
        else:
            b = np.nan

        out[t] = {'b': b, 'avg_win': avg_win, 'avg_loss_abs': avg_loss_abs}
    return out


# ##### PyTorch Helpers

# In[18]:


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class RNNClassifier(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, fc=32,
                 dropout=0.3, inter_dropout=0.1, num_layers=2, use_layernorm=False,
                 rnn_type='LSTM', bidirectional=False):
        super().__init__()
        self.num_layers = int(num_layers)
        if self.num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")

        self.rnn_type = str(rnn_type).upper()
        self.bidirectional = bool(bidirectional)
        rnn_cls = nn.GRU if 'GRU' in self.rnn_type else nn.LSTM

        self.rnn1 = rnn_cls(
            input_size=n_features, hidden_size=hidden1,
            num_layers=1, batch_first=True, bidirectional=self.bidirectional, dropout=0.1
        )
        self.inter_drop = nn.Dropout(p=inter_dropout)  # proxy for recurrent_dropout

        self.rnn2 = None
        if self.num_layers == 2:
            self.rnn2 = rnn_cls(
                input_size=hidden1 * (2 if self.bidirectional else 1), hidden_size=hidden2,
                num_layers=1, batch_first=True, bidirectional=self.bidirectional, dropout=0.1
            )
            feat_dim = hidden2 * (2 if self.bidirectional else 1)
        else:
            feat_dim = hidden1 * (2 if self.bidirectional else 1)

        # Normalization after temporal pooling
        self.use_layernorm = bool(use_layernorm)
        if self.use_layernorm:
            self.norm = nn.LayerNorm(normalized_shape=feat_dim)
        else:
            self.norm = nn.BatchNorm1d(num_features=feat_dim)

        # Head
        self.fc1 = nn.Linear(feat_dim, fc)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(fc, 1)  # logits

    def forward(self, x):
        # x: (B, T, F)
        out1, _ = self.rnn1(x)     # (B, T, hidden1*(1|2))
        if self.num_layers == 2:
            out1 = self.inter_drop(out1)
            out2, _ = self.rnn2(out1)  # (B, T, hidden2*(1|2))
            last = out2[:, -1, :]
        else:
            last = out1[:, -1, :]

        if isinstance(self.norm, nn.BatchNorm1d):
            last = self.norm(last)  # BN expects (B, C)
        else:
            last = self.norm(last)  # LN expects (B, C)

        z = self.fc1(last)
        z = self.relu(z)
        z = self.drop(z)
        logits = self.fc_out(z).squeeze(-1)  # (B,)
        return logits
def _model_key(model_type: str) -> str:
    return str(model_type or 'LSTM').strip().lower()


def _artifact_name(ticker: str, H: int, model_type: str, kind: str) -> str:
    key = _model_key(model_type)
    if key == 'lstm':
        if kind == 'state':
            return f"{ticker}_lstm_H{H}.pt"
        if kind == 'cal':
            return f"{ticker}_calibrator_H{H}.pkl"
        if kind == 'scaler':
            return f"{ticker}_scaler_H{H}.pkl"
        if kind == 'seq':
            return f"{ticker}_seq_length_H{H}.pkl"
    if kind == 'state':
        return f"{ticker}_{key}_H{H}.pt"
    if kind == 'cal':
        return f"{ticker}_calibrator_{key}_H{H}.pkl"
    if kind == 'scaler':
        return f"{ticker}_scaler_{key}_H{H}.pkl"
    if kind == 'seq':
        return f"{ticker}_seq_length_{key}_H{H}.pkl"
    raise ValueError(f"Unknown artifact kind: {kind}")

class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.best_state_dict = None
        self.mode = mode  # 'min' for val_loss

    def step(self, metric, model):
        if self.best_metric is None:
            improved = True
        else:
            if self.mode == 'min':
                improved = (self.best_metric - metric) > self.min_delta
            else:
                improved = (metric - self.best_metric) > self.min_delta
        if improved:
            self.best_metric = metric
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return improved

@torch.no_grad()
def _evaluate(model, loader, device, loss_fn, mixed_precision: bool = False):
    model.eval()
    total_loss = 0.0
    all_logits = []
    ctx = torch.amp.autocast('cuda') if (mixed_precision and device.type == 'cuda') else nullcontext()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with ctx:
            logits = model(xb)
            loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        all_logits.append(logits.detach().cpu())
    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    return avg_loss, probs


# #### Train Company Models
# Train, calibrate, and persist a per-ticker classifier.
# This encapsulates per-asset modelling with proper scaling and probability calibration, which is crucial since Kelly needs probabilities (not raw logits).

# In[19]:


from matplotlib import ticker


def train_company_models(company_data_df: pd.DataFrame,
                         ticker: str,
                         feature_cols: list,
                         model_save_path: str,
                         sequence_length: int = None,
                         H = 1,
                         cfg: dict | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixed_precision = (device.type == 'cuda' and USE_MIXED_PRECISION)
    amp_scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision)

    if cfg is None:
        cfg = globals().get('MODEL_CFG', {})

    # sequence length & guards
    if sequence_length is None:
        sequence_length = int(cfg.get('sequence_length', 0)) or calculate_dynamic_sequence_length(company_data_df)

    if len(company_data_df) < sequence_length + H + 10:
        print(f"Not enough data for {ticker}.")
        return False, sequence_length

    contiguous_periods = identify_contiguous_periods(company_data_df)
    if not contiguous_periods:
        print(f"No contiguous periods found for {ticker}.")
        return False, sequence_length


    # build sequences
    targets = company_data_df[f'target_{H}d'].values
    X, y = create_contiguous_sequences(
        company_data_df[feature_cols].values,
        targets,
        contiguous_periods,
        sequence_length,
        H
    )
    print(f"Created {len(X)} sequences for {ticker}.")
    if len(X) < 2:
        return False, sequence_length

    # chronological split (no shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # scale features (fit on train, transform both)
    scaler = MinMaxScaler()
    F = X_train.shape[-1]
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, F)).reshape(X_train.shape)
    X_val_scaled   = scaler.transform(  X_val.reshape(-1, F)).reshape(X_val.shape)

    print(f"[TRAIN] {ticker}: scaler fitted on {X_train.shape[0]} training sequences")
    # sanity checks
    assert F == len(feature_cols),         f"[TRAIN] {ticker}: feature dimension mismatch: X_train has {F}, feature_cols has {len(feature_cols)}"

    print(f"[TRAIN] {ticker}: scaler.n_features_in_ = {scaler.n_features_in_}, len(feature_cols) = {len(feature_cols)}")
    assert scaler.n_features_in_ == len(feature_cols),         f"[TRAIN] {ticker}: scaler expects {scaler.n_features_in_} features, but feature_cols has {len(feature_cols)}"

    # Keras fit used the scaled data
    train_ds = SequenceDataset(X_train_scaled, y_train)
    val_ds   = SequenceDataset(X_val_scaled,   y_val)

    batch_size = int(cfg.get('batch_size', 32))
    train_bs = min(batch_size, len(train_ds))
    if train_bs < 2:
        print(f"[TRAIN] {ticker}: too few samples for BatchNorm (train size={len(train_ds)})")
        return False, sequence_length
    if len(train_ds) % train_bs == 1 and train_bs > 2:
        train_bs -= 1  # avoid batch size 1 for BatchNorm
    val_bs = min(batch_size, len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=train_bs, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=val_bs, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS_EVAL, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS
    )

    n_features = X_train.shape[-1]
    hidden1 = int(cfg.get('hidden1', 128))
    hidden2 = int(cfg.get('hidden2', 64))
    num_layers = int(cfg.get('num_layers', 2))
    dropout = float(cfg.get('dropout', 0.3))
    inter_dropout = float(cfg.get('inter_rnn_drop', 0.1))
    model_type = str(cfg.get('model_type', 'LSTM'))
    rnn_type = 'GRU' if 'GRU' in model_type.upper() else 'LSTM'
    bidirectional = 'BI' in model_type.upper()

    model = RNNClassifier(
        n_features=n_features,
        hidden1=hidden1,
        hidden2=hidden2,
        num_layers=num_layers,
        dropout=dropout,
        inter_dropout=inter_dropout,
        rnn_type=rnn_type,
        bidirectional=bidirectional
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    learning_rate = float(cfg.get('learning_rate', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_factor = float(cfg.get('lr_factor', 0.5))
    lr_patience = int(cfg.get('lr_patience', 7))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=1e-7)

    max_epochs = int(cfg.get('max_epochs', 100))
    early_patience = int(cfg.get('early_patience', 15))
    early_min_delta = float(cfg.get('early_min_delta', 0.0))
    early = EarlyStopper(patience=early_patience, min_delta=early_min_delta, mode='min')

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast('cuda') if mixed_precision else nullcontext()
            with ctx:
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if mixed_precision:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clipnorm=1.0
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clipnorm=1.0
                optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, _ = _evaluate(model, val_loader, device, loss_fn, mixed_precision=mixed_precision)

        scheduler.step(val_loss)  # ReduceLROnPlateau on val_loss

        _ = early.step(val_loss, model)
        if early.counter >= early.patience:  # stop after 'patience' non-improve epochs
            break

    # restore best weights
    if early.best_state_dict is not None:
        model.load_state_dict(early.best_state_dict)

    # final validation pass for isotonic calibration (on scaled val data)
    _, validation_predictions = _evaluate(model, val_loader, device, loss_fn, mixed_precision=mixed_precision)

    pred_min, pred_max = validation_predictions.min(), validation_predictions.max()
    buffer = 0.05 * (pred_max - pred_min)
    y_min_dynamic = max(0.0, pred_min - buffer)
    y_max_dynamic = min(1.0, pred_max + buffer)

    calibrator = IsotonicRegression(y_min=y_min_dynamic, y_max=y_max_dynamic, out_of_bounds='clip')
    calibrator.fit(validation_predictions, y_val.astype(float))

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "state")))
    joblib.dump(calibrator,      os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "cal")))
    joblib.dump(scaler,          os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "scaler")))
    joblib.dump(sequence_length, os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "seq")))

    del model, scaler, calibrator, train_loader, val_loader, train_ds, val_ds
    del X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True, sequence_length


# #### Predict Next Day Performance 
# One-step-ahead inference for a ticker on a given day.
# 
# Feeds calibrated p(up) into the portfolio/Kelly step.

# In[ ]:


def predict_next_horizon(company_data_df: pd.DataFrame,
                                 ticker: str,
                                 feature_cols: list,
                                 model_save_path: str,
                                 H: int,
                                 cfg: dict | None = None) -> dict | None:

    if cfg is None:
        cfg = globals().get('MODEL_CFG', {})

    model_type = str(cfg.get('model_type', 'LSTM'))
    rnn_type = 'GRU' if 'GRU' in model_type.upper() else 'LSTM'
    bidirectional = 'BI' in model_type.upper()

    try:
        # load artifacts
        state_dict_path = os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "state"))
        calibrator = joblib.load(os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "cal")))
        scaler     = joblib.load(os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "scaler")))
        sequence_length    = joblib.load(os.path.join(model_save_path, _artifact_name(ticker, H, model_type, "seq")))

        if not os.path.isfile(state_dict_path):
            return None

        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(feature_cols):
            return None

        # build model skeleton with correct input dim, then load weights
        n_features = len(feature_cols)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RNNClassifier(
            n_features=n_features,
            hidden1=int(cfg.get('hidden1', 128)),
            hidden2=int(cfg.get('hidden2', 64)),
            num_layers=int(cfg.get('num_layers', 2)),
            dropout=float(cfg.get('dropout', 0.3)),
            inter_dropout=float(cfg.get('inter_rnn_drop', 0.1)),
            rnn_type=rnn_type,
            bidirectional=bidirectional
        ).to(device)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        model.eval()
    except Exception:
        return None

    # find the last contiguous block and take the most recent sequence_length rows
    contiguous_periods = identify_contiguous_periods(company_data_df)
    if not contiguous_periods:
        return None

    last_start, last_end = contiguous_periods[-1]
    period_data = company_data_df.iloc[last_start:last_end + 1]
    if len(period_data) < sequence_length:
        return None

    last_sequence = period_data.tail(sequence_length)

    # scale using the saved scaler (fit train only)
    try:
        scaled_features = scaler.transform(last_sequence[feature_cols])
    except Exception:
        return None

    # shape: (1, T, F) as float32 tensor
    x = torch.from_numpy(np.asarray(scaled_features, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        ctx = torch.amp.autocast('cuda') if (torch.cuda.is_available() and USE_MIXED_PRECISION) else nullcontext()
        with ctx:
            logits = model(x)
            prob = torch.sigmoid(logits).item()

    calibrated_prediction = calibrator.predict([prob])[0]

    return {
        'ticker': ticker,
        'predicted_prob': prob,
        'calibrated_prediction': calibrated_prediction
    }


# #### Select and Size Portfolio
# Turn a cross-section of calibrated predictions into position sizes.
# 
# Enforces diversification (one per sector) and risk discipline (Fractional Kelly).

# In[21]:


def select_and_size_portfolio(daily_predictions_df: pd.DataFrame, payout_map: dict,
                            total_capital: float, sector_threshold: float,
                            kelly_fraction: float, top_k: int = 3, softmax_alpha: float = 1) -> pd.DataFrame:

    investment_decisions = []

    if daily_predictions_df.empty:
        print("[PORT] No predictions provided today.")
        return pd.DataFrame()

    print(f"[PORT] Received {len(daily_predictions_df)} predictions for sizing.")

    # sector-level filter
    df = daily_predictions_df.copy()
    df = df.groupby('sector').filter(lambda g: g['calibrated_prediction'].mean() >= sector_threshold)
    if df.empty:
        print("[PORT] No sectors cleared confidence threshold.")
        return pd.DataFrame()

    candidates = []
    for _, row in df.iterrows():
        ticker = row['ticker']
        p = float(row['calibrated_prediction'])
        b_info = payout_map.get(ticker, 0.0)
        b = (b_info['b'] if isinstance(b_info, dict) else float(b_info))
        if not np.isfinite(b) or b <= 0:
            continue
        if not b_info or not np.isfinite(b_info.get('b', np.nan)):
            continue
        b = max(b_info['b'], 1e-6)  # avoid div-by-zero
        kelly_percentage = p - (1 - p) / b
        if kelly_percentage <= 0:
            continue
        investment_fraction = kelly_percentage * kelly_fraction
        investment_amount = total_capital * investment_fraction
        if investment_amount < MIN_TRADE_DOLLARS:
            continue

        avg_win = b_info.get('avg_win', np.nan) if isinstance(b_info, dict) else np.nan
        avg_loss = b_info.get('avg_loss_abs', np.nan) if isinstance(b_info, dict) else np.nan
        if np.isfinite(avg_win) and np.isfinite(avg_loss):
            expected_rise = p * avg_win - (1.0 - p) * avg_loss
        else:
            expected_rise = p

        cand = row.to_dict()
        cand.update({
            'kelly_percentage': kelly_percentage,
            'investment_fraction': investment_fraction,
            'investment_amount': investment_amount,
            'expected_rise': expected_rise,
            'prob_up': p,
        })
        candidates.append(cand)

    if not candidates:
        print("[PORT] All candidates rejected after sizing/filters.")
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates)

    rows = []
    alpha = max(softmax_alpha, 1e-6)
    for sector, group in cand_df.groupby('sector'):
        if group.empty:
            continue
        top_group = group.sort_values('calibrated_prediction', ascending=False).head(max(1, top_k))
        logits = top_group['calibrated_prediction'].to_numpy() * 100.0 * alpha
        logits = logits - logits.max()  # stabilize
        weights = np.exp(logits)
        probs = weights / weights.sum()
        choice_idx = np.random.choice(len(top_group), p=probs)
        choice = top_group.iloc[choice_idx]

        print(f"[PORT] Sector {sector}: sampled {choice['ticker']} from top-{len(top_group)} (p={choice['calibrated_prediction']:.3f})")

        rows.append({
            'ticker': choice['ticker'],
            'sector': choice.get('sector', sector),
            'expected_rise': choice.get('expected_rise', np.nan),
            'prob_up': choice.get('prob_up', np.nan),
            'investment_fraction': choice['investment_fraction'],
            'investment_amount': choice['investment_amount'],
            'predicted_prob': choice['calibrated_prediction']
        })

    if not rows:
        print("[PORT] All sector winners rejected after sampling.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# #### Models Exist for Ticker
# Quick guard to avoid retraining if a ticker’s artifacts already exist.
# 
# Decide whether to train.

# In[22]:


def models_exist_for_ticker(ticker: str, model_path: str, H: int = 1, model_type: str = 'LSTM') -> bool:
    files = [
        _artifact_name(ticker, H, model_type, 'state'),
        _artifact_name(ticker, H, model_type, 'cal'),
        _artifact_name(ticker, H, model_type, 'scaler'),
        _artifact_name(ticker, H, model_type, 'seq'),
    ]
    return all(os.path.exists(os.path.join(model_path, f)) for f in files)


# #### Run Simulation
# The “engine” — walks forward over dates, trains as needed, predicts, sizes, books PnL, and updates capital.
# 
# Enforces chronological integrity, periodic retraining, diversification, and proper capital tracking.

# In[23]:


def _project_to_simplex(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    if np.all(v >= 0) and np.isclose(v.sum(), 1.0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.ones_like(v) / len(v)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    return w


def _apply_max_weight_cap(w, max_w, tol=1e-8):
    if max_w is None:
        return w
    w = np.maximum(w, 0.0)
    if w.size == 0:
        return w
    cap = float(max_w)
    w = np.minimum(w, cap)
    total = w.sum()
    if total <= 0:
        return w
    if total > 1.0 + tol:
        w = _project_to_simplex(w)
        w = np.minimum(w, cap)
    if total < 1.0 - tol:
        remaining = 1.0 - w.sum()
        for _ in range(100):
            room = cap - w
            room[room < 0] = 0.0
            if room.sum() <= tol:
                break
            add = remaining * (room / room.sum())
            w += add
            remaining = 1.0 - w.sum()
            if abs(remaining) <= tol:
                break
    if w.sum() > 0:
        w = w / w.sum()
    return w


def compute_mpt_weights(mu, Sigma, risk_lambda=1.0, long_only=True,
                        ridge_eps=1e-4, max_weight=None,
                        use_equal_fallback=True):
    mu = np.asarray(mu, dtype=np.float64)
    n = mu.size
    if n == 0:
        return mu
    if Sigma is None:
        return np.ones(n) / n if use_equal_fallback else mu / np.maximum(mu.sum(), 1e-12)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if Sigma.shape != (n, n):
        return np.ones(n) / n if use_equal_fallback else mu / np.maximum(mu.sum(), 1e-12)

    Sigma_reg = Sigma + ridge_eps * np.eye(n)

    def _fallback():
        w = np.ones(n) / n if use_equal_fallback else np.maximum(mu, 0.0)
        w = w / np.maximum(w.sum(), 1e-12)
        return _apply_max_weight_cap(w, max_weight)

    try:
        try:
            import scipy.optimize as opt
            def obj(w):
                return -(mu @ w - risk_lambda * (w @ Sigma_reg @ w))
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
            bounds = None
            if long_only:
                upper = max_weight if max_weight is not None else 1.0
                bounds = [(0.0, upper) for _ in range(n)]
            x0 = np.ones(n) / n
            res = opt.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
            if not res.success or not np.isfinite(res.x).all():
                raise ValueError('MPT optimization failed')
            w = res.x
        except Exception:
            inv = np.linalg.pinv(Sigma_reg)
            w = inv @ mu
            if long_only:
                w = np.maximum(w, 0.0)
            if w.sum() <= 0:
                return _fallback()
            w = w / w.sum()
        if long_only:
            w = _apply_max_weight_cap(w, max_weight)
        if not np.isfinite(w).all() or w.sum() <= 0:
            return _fallback()
        return w
    except Exception:
        return _fallback()


def build_covariance_matrix(master_df, tickers, end_date, window_days, min_obs=20):
    df = master_df[master_df['date'] < end_date].copy()
    if df.empty:
        return None, []
    df = df[df['ticker'].isin(tickers)].copy()
    if df.empty:
        return None, []
    df['date'] = pd.to_datetime(df['date'])
    unique_dates = sorted(df['date'].unique())
    if not unique_dates:
        return None, []
    if len(unique_dates) > window_days:
        unique_dates = unique_dates[-window_days:]
    df = df[df['date'].isin(unique_dates)]
    pivot = df.pivot_table(index='date', columns='ticker', values='close', aggfunc='last').sort_index()
    rets = pivot.pct_change().dropna(how='all')
    if rets.empty:
        return None, []
    min_obs = max(5, min_obs)
    good_cols = [c for c in rets.columns if rets[c].count() >= min_obs]
    rets = rets[good_cols]
    if rets.shape[0] < min_obs or rets.shape[1] == 0:
        return None, []
    Sigma = rets.cov().values
    return Sigma, list(rets.columns)


# In[24]:


def compute_mpt_weights_for_today(
    tickers,
    historical_data,
    current_date,
    mu,
    window_days=MPT_WINDOW_DAYS,
    ridge_eps=MPT_RIDGE_EPS,
    risk_aversion=MPT_LAMBDA,
    long_only=MPT_LONG_ONLY,
    max_weight=MPT_MAX_WEIGHT,
    use_equal_fallback=MPT_USE_EQUAL_FALLBACK,
):
    if len(tickers) == 0:
        return np.array([]), True

    Sigma, cov_tickers = build_covariance_matrix(
        historical_data,
        tickers,
        current_date,
        window_days=window_days,
        min_obs=20,
    )
    if Sigma is None or set(cov_tickers) != set(tickers):
        w = np.ones(len(tickers)) / len(tickers)
        return w, True

    order = [cov_tickers.index(t) for t in tickers]
    Sigma = Sigma[np.ix_(order, order)]

    w = compute_mpt_weights(
        mu,
        Sigma,
        risk_lambda=risk_aversion,
        long_only=long_only,
        ridge_eps=ridge_eps,
        max_weight=max_weight,
        use_equal_fallback=use_equal_fallback,
    )
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(tickers)) / len(tickers)
        return w, True
    return w, False


def apply_allocation_mode(
    investment_df,
    historical_data,
    current_date,
    capital,
    open_positions,
    allocation_mode=ALLOCATION_MODE,
):
    meta = {
        'allocation_mode': allocation_mode,
        'budget_pre_caps': 0.0,
        'mpt_used': False,
        'mpt_fallback_used': False,
        'weights': None,
    }

    if investment_df is None or investment_df.empty:
        return investment_df, meta

    mode = str(allocation_mode).upper().strip()
    if mode not in {"KELLY_ONLY", "HYBRID_KELLY_MPT", "MPT_ONLY"}:
        mode = "KELLY_ONLY"
        meta['allocation_mode'] = mode

    budget_pre_caps = float(investment_df['investment_amount'].sum())
    tickers = investment_df['ticker'].tolist()

    if mode == "KELLY_ONLY":
        meta['budget_pre_caps'] = budget_pre_caps
        return investment_df, meta

    if mode == "MPT_ONLY":
        equity = float(capital + sum(p['invest'] for p in open_positions))
        budget_pre_caps = float(MPT_ONLY_BUDGET_FRACTION * equity)

    meta['budget_pre_caps'] = budget_pre_caps

    if len(tickers) == 1:
        if mode == "MPT_ONLY":
            meta['mpt_used'] = True
            meta['weights'] = {tickers[0]: 1.0}
            investment_df = investment_df.copy()
            investment_df['investment_amount'] = budget_pre_caps
        return investment_df, meta

    mu = investment_df['expected_rise'].astype(float).values
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    w, fallback = compute_mpt_weights_for_today(
        tickers,
        historical_data,
        current_date,
        mu,
    )
    meta['mpt_used'] = True
    meta['mpt_fallback_used'] = bool(fallback)

    if np.isfinite(w).all() and w.sum() > 0:
        w = w / w.sum()
        meta['weights'] = {t: float(w[i]) for i, t in enumerate(tickers)}
        investment_df = investment_df.copy()
        investment_df['investment_amount'] = budget_pre_caps * w
        return investment_df, meta

    return investment_df, meta


# In[ ]:


def run_simulation(master_df: pd.DataFrame, feature_cols: list,
                   initial_capital: float, initial_training_days: int = INITIAL_TRAINING_DAYS,
                   H: int = 1, allow_overlap: bool = True, early_exit_enabled: bool = True):

    print(f"Starting simulation with {initial_training_days} initial training days... (H={H})")
    cfg = globals().get('MODEL_CFG', {})
    if cfg:
        keys = ['model_type','feature_set','feature_group','sequence_length','hidden1','hidden2','num_layers',
                'dropout','inter_rnn_drop','batch_size','learning_rate','weight_decay','ordinal_head','n_classes','horizon_steps']
        parts = []
        for k in keys:
            if k in cfg and cfg[k] is not None:
                parts.append(f"{k}={cfg[k]}")
        parts.append(f"horizon={H}")
        print("[CONFIG] " + ', '.join(parts))
    else:
        print("[CONFIG] MODEL_CFG not set")
    capital = initial_capital
    equity_curve = [capital]
    simulation_log = []

    ret_col = f"ret_{H}d"
    tgt_col = f"target_{H}d"

    # returns used for settlement only; not for training
    master_df_returns = add_horizon_targets(master_df.copy(), H=H, price_col='close')
    ret_lookup = master_df_returns.set_index(['ticker','date'])[ret_col].to_dict()
    close_lookup = master_df.set_index(['ticker', 'date'])['close'].to_dict()

    sectors_by_ticker = master_df.groupby('ticker')['sector'].first().to_dict()

    all_tickers = master_df['ticker'].unique()
    retrain_counter = {ticker: 0 for ticker in all_tickers}
    ticker_sequence_lengths = {}

    model_type = None
    if 'MODEL_CFG' in globals():
        model_type = MODEL_CFG.get('model_type')
    if not model_type:
        model_type = 'LSTM'

    early_exit_lookup = None
    early_exit_threshold = 0.5  # exit when p_up_1d < threshold (downside signal)
    early_exit_model = None
    if 'MODEL_CFG' in globals():
        early_exit_model = MODEL_CFG.get('model_type')
    if not early_exit_model:
        early_exit_model = 'LSTM'
    if early_exit_enabled and H > 1:
        early_exit_path = EARLY_EXIT_PATH
        if not os.path.exists(early_exit_path):
            early_exit_path = EARLY_EXIT_PATH
        if os.path.exists(early_exit_path):
            try:
                h1_exit_df = pd.read_parquet(early_exit_path)
                h1_exit_df['date'] = pd.to_datetime(h1_exit_df['date'])
                h1_exit_df = h1_exit_df.drop_duplicates(subset=['date', 'ticker'])

                prob_col = f"p_up_1d_{early_exit_model}"
                dir_col  = f"reg_dir_{early_exit_model}"
                conf_col = f"reg_conf_{early_exit_model}"
                if prob_col not in h1_exit_df.columns:
                    prob_col = 'p_up_1d' if 'p_up_1d' in h1_exit_df.columns else None
                if dir_col not in h1_exit_df.columns:
                    dir_col = None
                if conf_col not in h1_exit_df.columns:
                    conf_col = None

                if prob_col is None or dir_col is None or conf_col is None:
                    print(f"[EARLY EXIT] Missing columns for model={early_exit_model}. prob_col={prob_col}, dir_col={dir_col}. Early exit disabled.")
                    early_exit_lookup = None
                else:
                    early_exit_lookup = h1_exit_df.set_index(['date', 'ticker'])[[prob_col, dir_col, conf_col]].to_dict('index')
                    print(f"[EARLY EXIT] Loaded {len(h1_exit_df)} signals from {early_exit_path} using {prob_col} + {dir_col}")
            except Exception as exc:
                print(f"[EARLY EXIT] Failed to load signals: {exc}")
                early_exit_lookup = None
        else:
            print("[EARLY EXIT] No 1d exit signal file found; early exit disabled.")

    unique_dates = sorted(master_df['date'].unique())
    start_index = min(initial_training_days, len(unique_dates) - 1)
    print(f"Starting predictions from day {start_index} (after {initial_training_days} training days)")

    open_positions = []
    prev_equity_end = capital
    total_early_exits = 0
    early_exit_pnl_total = 0.0
    early_exit_counterfactual_total = 0.0
    early_exit_counterfactual_count = 0

    for i in tqdm(range(start_index, len(unique_dates)), desc="Simulating Trading Days"):
        current_date = pd.to_datetime(unique_dates[i])
        prev_date = pd.to_datetime(unique_dates[i-1]) if i > 0 else None
        early_exits_today = 0
        exits_today = []

        # settle positions exiting today
        total_pnl_today = 0.0
        still_open = []
        for pos in open_positions:
            if pos['exit_date'] == current_date:
                rH = ret_lookup.get((pos['ticker'], pos['entry_date']), np.nan)
                entry_close = close_lookup.get((pos['ticker'], pos['entry_date']))
                exit_close = close_lookup.get((pos['ticker'], current_date))
                r_exit = np.nan
                if entry_close is not None and exit_close is not None and np.isfinite(entry_close) and np.isfinite(exit_close):
                    r_exit = (exit_close / entry_close) - 1.0
                elif pd.notna(rH):
                    r_exit = rH
                if pd.notna(r_exit):
                    pnl = pos['invest'] * r_exit
                    total_pnl_today += pnl
                    capital += pos['invest'] + pnl
                    exits_today.append({
                        'date': pos['entry_date'],
                        'exit_date': current_date,
                        'ticker': pos['ticker'],
                        'sector': pos.get('sector', sectors_by_ticker.get(pos['ticker'], 'UNKNOWN')),
                        'allocated_amount': pos['invest'],
                        'prob_up': pos.get('prob_up', np.nan),
                        'expected_rise': pos.get('expected_rise', np.nan),
                        'actual_return': r_exit,
                        'pnl': pnl,
                        'exit_reason': 'scheduled',
                    })
            else:
                still_open.append(pos)
        open_positions = still_open

        # early-exit check (signal at current_date predicts tomorrow)
        if early_exit_enabled and H > 1 and early_exit_lookup:
            remaining = []
            for pos in open_positions:
                sig = early_exit_lookup.get((current_date, pos['ticker']))
                if sig is None:
                    remaining.append(pos)
                    continue
                p_up_1d = sig.get(prob_col) if isinstance(sig, dict) else None
                reg_dir = sig.get(dir_col) if isinstance(sig, dict) else None
                reg_conf = sig.get(conf_col) if isinstance(sig, dict) else None
                if p_up_1d is None or reg_dir is None or reg_conf is None or not np.isfinite(p_up_1d):
                    remaining.append(pos)
                    continue
                if float(p_up_1d) < early_exit_threshold and (int(reg_dir) == -1 and float(reg_conf) > 0.1):
                    entry_close = close_lookup.get((pos['ticker'], pos['entry_date']))
                    exit_close = close_lookup.get((pos['ticker'], current_date))
                    if entry_close is None or exit_close is None:
                        remaining.append(pos)
                        continue
                    if not np.isfinite(entry_close) or not np.isfinite(exit_close):
                        remaining.append(pos)
                        continue
                    r_exit = (exit_close / entry_close) - 1.0
                    pnl = pos['invest'] * r_exit
                    total_pnl_today += pnl
                    capital += pos['invest'] + pnl
                    early_exits_today += 1
                    total_early_exits += 1
                    early_exit_pnl_total += pnl

                    planned_rH = ret_lookup.get((pos['ticker'], pos['entry_date']), np.nan)
                    if pd.notna(planned_rH):
                        early_exit_counterfactual_total += pos['invest'] * planned_rH
                        early_exit_counterfactual_count += 1

                    exits_today.append({
                        'date': pos['entry_date'],
                        'exit_date': current_date,
                        'ticker': pos['ticker'],
                        'sector': pos.get('sector', sectors_by_ticker.get(pos['ticker'], 'UNKNOWN')),
                        'allocated_amount': pos['invest'],
                        'prob_up': pos.get('prob_up', np.nan),
                        'expected_rise': pos.get('expected_rise', np.nan),
                        'actual_return': r_exit,
                        'pnl': pnl,
                        'exit_reason': 'early_exit',
                    })

                    if VERBOSE:
                        print(
                            f"[EARLY EXIT] {current_date}: {pos['ticker']} p_up_1d={p_up_1d:.3f} -> exit; "
                            f"r={r_exit:.4f}, PnL=${pnl:.2f}"
                        )
                else:
                    remaining.append(pos)
            open_positions = remaining

        open_notional_start = sum(p['invest'] for p in open_positions)
        equity_start = capital + open_notional_start
        daily_return = (equity_start / prev_equity_end) - 1.0 if prev_equity_end > 0 else 0.0

        # historical slice strictly before current_date; recompute targets locally
        historical_data = master_df[master_df['date'] < current_date].copy()
        historical_data = historical_data.drop(columns=[ret_col, tgt_col], errors='ignore')
        if historical_data.empty:
            continue
        historical_data = add_horizon_targets(historical_data, H=H, price_col='close')
        historical_data = historical_data.dropna(subset=[ret_col])
        cutoff = current_date - pd.Timedelta(days=H)
        historical_data = historical_data[historical_data['date'] < cutoff]
        if historical_data.empty:
            continue
        payout_map_day = calculate_historical_payouts(historical_data, ret_col=ret_col)

        if prev_date is None:
            todays_data_for_prediction = pd.DataFrame(columns=master_df.columns)
        else:
            todays_data_for_prediction = master_df[master_df['date'] == prev_date]

        daily_predictions = []
        for ticker in todays_data_for_prediction['ticker'].unique():
            company_hist_data = historical_data[historical_data['ticker'] == ticker]
            if company_hist_data.empty:
                continue

            need_retrain = retrain_counter.get(ticker, 0) >= RETRAIN_INTERVAL
            have_models  = models_exist_for_ticker(ticker, MODEL_SAVE_PATH, H=H, model_type=model_type)
            if not have_models or need_retrain:
                training_success, seq_length = train_company_models(
                    company_hist_data, ticker, feature_cols, MODEL_SAVE_PATH, H=H
                )
                if training_success:
                    ticker_sequence_lengths[ticker] = seq_length
                    retrain_counter[ticker] = 0
                else:
                    continue

            pred = predict_next_horizon(company_hist_data, ticker, feature_cols, MODEL_SAVE_PATH, H=H)
            if pred:
                info = todays_data_for_prediction[todays_data_for_prediction['ticker'] == ticker].iloc[0]
                pred.update({'company_name': info['company_name'], 'sector': info['sector']})
                daily_predictions.append(pred)
                retrain_counter[ticker] += 1

        daily_predictions_df = pd.DataFrame(daily_predictions)

        investment_decision_df = select_and_size_portfolio(
            daily_predictions_df, payout_map_day, capital, SECTOR_CONFIDENCE_THRESHOLD, KELLY_FRACTION
        )

        investment_decision_df, alloc_meta = apply_allocation_mode(
            investment_decision_df,
            historical_data,
            current_date,
            capital,
            open_positions,
            allocation_mode=ALLOCATION_MODE,
        )

        exit_idx = i + H
        exit_date = pd.to_datetime(unique_dates[exit_idx]) if exit_idx < len(unique_dates) else None

        entries_today = []
        if exit_date is not None and not investment_decision_df.empty:
            investment_decision_df = investment_decision_df.sort_values('predicted_prob', ascending=False)

            util_mult = utilization_throttle(equity_curve)
            total_open, expo_ticker, expo_sector = compute_exposures(open_positions, sectors_by_ticker)
            new_today_open = 0.0
            budget_post_caps = 0.0

            for _, trade in investment_decision_df.iterrows():
                tkr = trade['ticker']
                sec = sectors_by_ticker.get(tkr, 'UNKNOWN')

                open_count = sum(1 for p in open_positions if p['ticker'] == tkr)
                if allow_overlap:
                    if open_count >= MAX_POSITIONS_PER_TICKER:
                        continue
                else:
                    if open_count >= 1:
                        continue

                base_amt = float(trade['investment_amount'])
                base_amt = min(base_amt, KELLY_TRADE_CAP * (capital + sum(p['invest'] for p in open_positions)))

                amt = clamp_by_caps(base_amt, (capital + sum(p['invest'] for p in open_positions)),
                                    total_open, new_today_open, expo_ticker, expo_sector,
                                    tkr, sec, util_mult)
                if amt < MIN_TRADE_DOLLARS:
                    continue

                capital -= amt
                total_open += amt
                new_today_open += amt
                budget_post_caps += amt
                expo_ticker[tkr] = expo_ticker.get(tkr, 0.0) + amt
                expo_sector[sec] = expo_sector.get(sec, 0.0) + amt

                pos = {
                    'ticker': tkr,
                    'invest': amt,
                    'entry_date': current_date,
                    'exit_date': exit_date,
                    'sector': sec,
                    'prob_up': float(trade.get('prob_up', trade.get('predicted_prob', np.nan))),
                    'expected_rise': float(trade.get('expected_rise', np.nan)),
                }
                open_positions.append(pos)
                entries_today.append({**trade.to_dict(), 'allocated_amount': amt, 'exit_date': exit_date})

        open_notional_end = sum(p['invest'] for p in open_positions)
        equity_end = capital + open_notional_end

        simulation_log.append({
            'date': current_date,
            'capital_start': equity_start - open_notional_start,
            'capital_end': capital,
            'equity_start': equity_start,
            'equity_end': equity_end,
            'daily_pnl_realized': total_pnl_today,
            'daily_return': daily_return,
            'early_exits_today': early_exits_today,
            'allocation_mode': alloc_meta['allocation_mode'],
            'budget_pre_caps': float(alloc_meta['budget_pre_caps']),
            'budget_post_caps': float(budget_post_caps) if 'budget_post_caps' in locals() else 0.0,
            'mpt_used': bool(alloc_meta['mpt_used']),
            'mpt_fallback_used': bool(alloc_meta['mpt_fallback_used']),
            'mpt_weights': alloc_meta.get('weights'),
            'exits_today': exits_today,
            'investments_made': entries_today
        })

        equity_curve.append(equity_end)
        prev_equity_end = equity_end

    print(f"[EARLY EXIT] Total early exits: {total_early_exits}")
    if total_early_exits > 0:
        print(f"[EARLY EXIT] PnL realized from early exits: ${early_exit_pnl_total:,.2f}")
        if early_exit_counterfactual_count > 0:
            print(
                f"[EARLY EXIT] PnL if held to horizon: ${early_exit_counterfactual_total:,.2f} "
                f"(n={early_exit_counterfactual_count})"
            )
            print(
                f"[EARLY EXIT] Delta vs hold-to-horizon: "
                f"${early_exit_pnl_total - early_exit_counterfactual_total:,.2f}"
            )
        else:
            print("[EARLY EXIT] Counterfactual PnL unavailable for early exits.")
    return pd.DataFrame(simulation_log)


# #### Calculate Final Results
# 

# In[26]:


def calculate_final_results(simulation_log: pd.DataFrame, initial_capital: float, rf_annual: float = 0.0):
    if simulation_log.empty:
        print("Simulation log is empty. No results to calculate.")
        return

    use_equity = {'equity_start', 'equity_end'}.issubset(simulation_log.columns)

    # precomputed daily_return if available and sane
    if 'daily_return' in simulation_log.columns:
        daily_returns = simulation_log['daily_return'].astype(float).values
    elif use_equity:
        eq_start = simulation_log['equity_start'].astype(float).values
        eq_end   = simulation_log['equity_end'].astype(float).values
        # r_t = equity_start_t / equity_end_{t-1} - 1
        prev_end = np.roll(eq_end, 1)
        prev_end[0] = eq_end[0] if eq_end[0] != 0 else eq_start[0]
        daily_returns = (eq_start / prev_end) - 1.0
    else:
        # cash-based (not ideal for H>0)
        print("Warning: Using cash-based returns")
        cap_start = simulation_log['capital_start'].astype(float).values
        cap_end   = simulation_log['capital_end'].astype(float).values
        prev_end  = np.roll(cap_end, 1)
        prev_end[0] = cap_end[0] if cap_end[0] != 0 else cap_start[0]
        daily_returns = (cap_start / prev_end) - 1.0

    daily_returns = np.where(np.isfinite(daily_returns), daily_returns, 0.0)

    # ROI from final equity if present, else cash
    end_col = 'equity_end' if use_equity else 'capital_end'
    final_value = float(simulation_log[end_col].iloc[-1])
    total_roi = (final_value / initial_capital) - 1.0

    # Sharpe
    excess = daily_returns - (rf_annual / 252.0)
    vol = excess.std(ddof=1)
    sharpe_ratio = (excess.mean() / vol * np.sqrt(252.0)) if vol > 1e-12 else 0.0

    print("\n--- Simulation Results ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital:   ${final_value:,.2f}")
    print(f"Total Return on Investment (ROI): {total_roi:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print("--------------------------")


# ### Load Data

# In[27]:


master_df = pd.read_parquet(MASTER_PATH)


# In[28]:


master_df


# In[29]:


columns_to_check = [

    # Sentiment (single)
    'sentiment',

    # Emotions
    'emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy',
    'emotion_neutral', 'emotion_sadness', 'emotion_surprize',
    'emotion_anger_pct', 'emotion_disgust_pct', 'emotion_fear_pct',
    'emotion_joy_pct', 'emotion_neutral_pct', 'emotion_sadness_pct',
    'emotion_surprize_pct',

    # Unified emotion
    'positive_emotion', 'negative_emotion', 'uncertainty_emotion',
    'positive_emotion_pct', 'negative_emotion_pct', 'uncertainty_emotion_pct',

    # Stance
    'stance_label', 'stance_score',

    # FinBERT
    'finbert_label', 'finbert_score', 'finbert_up', 'finbert_down',
    'finbert_neutral',

    # Sector aggregates
    'sector_open_mean', 'sector_high_mean', 'sector_low_mean', 'sector_close_mean',
    'sector_volume_mean', 'sector_ret_1d', 'sector_ret_5d',
    'sector_ret_20d', 'sector_range', 'sector_vol_20d',
    'ema_12_sector', 'ema_26_sector', 'ema_50_sector', 'macd_12_26_9_sector',
    'macdh_12_26_9_sector', 'macds_12_26_9_sector', 'rsi_14_sector',
    'sector_bb_upper', 'sector_bb_middle', 'sector_bb_lower',
    'market_close', 'sector_rel_strength', 'sector_dispersion_1d',

    # Meta regression predictions
    'reg_pred_ret_1d_LSTM', 'reg_pred_ret_1d_BiLSTM', 'reg_pred_ret_1d_GRU',
    'reg_pred_ret_1d_BiGRU',

    # Regression errors/diagnostics
    'reg_abs_err_lag1_LSTM', 'reg_mae_20_LSTM',
    'reg_rmse_20_LSTM', 'reg_dir_acc_20_LSTM',

    'reg_abs_err_lag1_BiLSTM',
    'reg_mae_20_BiLSTM', 'reg_rmse_20_BiLSTM', 'reg_dir_acc_20_BiLSTM',

    'reg_abs_err_lag1_GRU',
    'reg_mae_20_GRU', 'reg_rmse_20_GRU', 'reg_dir_acc_20_GRU',

    'reg_abs_err_lag1_BiGRU',
    'reg_mae_20_BiGRU', 'reg_rmse_20_BiGRU', 'reg_dir_acc_20_BiGRU',

    # Meta classification predictions
    'cls_prob_up_1d_LSTM', 'cls_prob_up_1d_BiLSTM', 'cls_prob_up_1d_GRU',
    'cls_prob_up_1d_BiGRU',

    # Classification diagnostics
    'cls_brier_20_LSTM', 'cls_logloss_20_LSTM', 'cls_acc_20_LSTM',

    'cls_brier_20_BiLSTM', 'cls_logloss_20_BiLSTM', 'cls_acc_20_BiLSTM',

    'cls_brier_20_GRU', 'cls_logloss_20_GRU', 'cls_acc_20_GRU',

    'cls_brier_20_BiGRU', 'cls_logloss_20_BiGRU', 'cls_acc_20_BiGRU',
]

print(f"Initial master_df shape: {master_df.shape}")

master_df = master_df.dropna(subset=columns_to_check)

print(f"After dropping NaNs in selected columns, master_df shape: {master_df.shape}")

master_df.reset_index(drop=True, inplace=True)

print(master_df)


# In[30]:


feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'roll_ret_1d', 'roll_ret_5d', 
    'roll_ret_20d',
    
    'ema_12', 'ema_26', 'ema_50', 'macd_12_26_9', 'macdh_12_26_9',
    'macds_12_26_9', 'rsi_14', 'stochrsik_14_14_3_3', 'stochrsid_14_14_3_3',
    'atrr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'obv',
]


# In[31]:


print(master_df.shape)
master_df = master_df.dropna(subset=feature_columns).sort_values(['ticker','date'])
print(master_df.shape)


# In[ ]:


# Select best 5H config from benchmarking and align feature set
from pathlib import Path

sentiment_columns = [
    'sentiment',
]

emotion_columns = [
    'emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy',
    'emotion_neutral', 'emotion_sadness', 'emotion_surprize',
    'emotion_anger_pct', 'emotion_disgust_pct', 'emotion_fear_pct',
    'emotion_joy_pct', 'emotion_neutral_pct', 'emotion_sadness_pct',
    'emotion_surprize_pct',
]

unified_emotion_columns = [
    'positive_emotion', 'negative_emotion','uncertainty_emotion',
    'positive_emotion_pct', 'negative_emotion_pct','uncertainty_emotion_pct',
]

stance_columns = [
    'stance_label', 'stance_score',
]

finbert_columns = [
    'finbert_label', 'finbert_score', 'finbert_up', 'finbert_down',
    'finbert_neutral',
]

sector_columns = ['sector_open_mean', 'sector_high_mean', 'sector_low_mean', 'sector_close_mean',
                  'sector_volume_mean', 'sector_ret_1d', 'sector_ret_5d',
                  'sector_ret_20d', 'sector_range', 'sector_vol_20d', 
                  
                  'ema_12_sector','ema_26_sector', 'ema_50_sector', 'macd_12_26_9_sector',
                  'macdh_12_26_9_sector', 'macds_12_26_9_sector', 'rsi_14_sector',
                  'sector_bb_upper', 'sector_bb_middle', 'sector_bb_lower',
                  'market_close', 'sector_rel_strength', 'sector_dispersion_1d']

meta_regression_columns = [
    'reg_pred_ret_1d_LSTM', 'reg_pred_ret_1d_BiLSTM', 'reg_pred_ret_1d_GRU',
    'reg_pred_ret_1d_BiGRU',
]

meta_classification_columns = [
    'cls_prob_up_1d_LSTM', 'cls_prob_up_1d_BiLSTM', 'cls_prob_up_1d_GRU',
    'cls_prob_up_1d_BiGRU',
]

meta_regression_err_columns = [
    'reg_abs_err_lag1_LSTM', 'reg_mae_20_LSTM',
    'reg_rmse_20_LSTM', 'reg_dir_acc_20_LSTM',

    'reg_abs_err_lag1_BiLSTM',
    'reg_mae_20_BiLSTM', 'reg_rmse_20_BiLSTM', 'reg_dir_acc_20_BiLSTM',

    'reg_abs_err_lag1_GRU',
    'reg_mae_20_GRU', 'reg_rmse_20_GRU', 'reg_dir_acc_20_GRU',

    'reg_abs_err_lag1_BiGRU',
    'reg_mae_20_BiGRU', 'reg_rmse_20_BiGRU', 'reg_dir_acc_20_BiGRU',]

meta_classification_err_columns = [
    'cls_brier_20_LSTM', 'cls_logloss_20_LSTM', 'cls_acc_20_LSTM',

    'cls_brier_20_BiLSTM', 'cls_logloss_20_BiLSTM', 'cls_acc_20_BiLSTM',

    'cls_brier_20_GRU', 'cls_logloss_20_GRU', 'cls_acc_20_GRU',

    'cls_brier_20_BiGRU', 'cls_logloss_20_BiGRU', 'cls_acc_20_BiGRU',
]


BASE_FEATURE_COLUMNS = feature_columns.copy()

feature_sets = {
    'base': BASE_FEATURE_COLUMNS,
    'sentiment': BASE_FEATURE_COLUMNS + sentiment_columns,
    'emotion': BASE_FEATURE_COLUMNS + emotion_columns,
    'unified_emotion': BASE_FEATURE_COLUMNS + unified_emotion_columns,
    'finbert': BASE_FEATURE_COLUMNS + finbert_columns,
    'all_nlp': BASE_FEATURE_COLUMNS + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    'sector': BASE_FEATURE_COLUMNS + sector_columns,
    'sector_sentiment': BASE_FEATURE_COLUMNS + sector_columns + sentiment_columns,
    'sector_emotion': BASE_FEATURE_COLUMNS + sector_columns + emotion_columns,
    'sector_unified_emotion': BASE_FEATURE_COLUMNS + sector_columns + unified_emotion_columns,
    'sector_finbert': BASE_FEATURE_COLUMNS + sector_columns + finbert_columns,
    'sector_all_nlp': BASE_FEATURE_COLUMNS + sector_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    
    'meta': feature_columns + meta_regression_columns + meta_classification_columns,
    
    'meta_err': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns,
    
    'meta_sentiment': feature_columns + meta_regression_columns + meta_classification_columns + sentiment_columns,
    'meta_stance': feature_columns + meta_regression_columns + meta_classification_columns + stance_columns,
    'meta_emotion': feature_columns + meta_regression_columns + meta_classification_columns + emotion_columns,
    'meta_unified_emotion': feature_columns + meta_regression_columns + meta_classification_columns + unified_emotion_columns,
    'meta_finbert': feature_columns + meta_regression_columns + meta_classification_columns + finbert_columns,
    'meta_all_nlp': feature_columns + meta_regression_columns + meta_classification_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    'meta_sector': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns,
    'meta_sector_sentiment': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns + sentiment_columns,
    'meta_sector_emotion': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns + emotion_columns,
    'meta_sector_unified_emotion': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns + unified_emotion_columns,
    'meta_sector_finbert': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns + finbert_columns,
    'meta_sector_all_nlp': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    
    'meta_err_sentiment': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sentiment_columns,
    'meta_err_stance': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + stance_columns,
    'meta_err_emotion': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + emotion_columns,
    'meta_err_unified_emotion': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + unified_emotion_columns,
    'meta_err_finbert': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + finbert_columns,
    'meta_err_all_nlp': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns, 
    'meta_err_sector': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns,
    'meta_err_sector_sentiment': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + sentiment_columns,
    'meta_err_sector_emotion': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + emotion_columns,
    'meta_err_sector_unified_emotion': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + unified_emotion_columns,
    'meta_err_sector_finbert': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + finbert_columns,
    'meta_err_sector_all_nlp': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,    
}


# In[33]:


def _dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _select_feature_cols(feature_set_name: str):
    name = FEATURE_SET_ALIAS.get(feature_set_name, feature_set_name)
    cols = feature_sets.get(name, BASE_FEATURE_COLUMNS)
    cols = _dedupe(cols)
    missing = [c for c in cols if c not in master_df.columns]
    if missing:
        print(f"[WARN] Missing columns for feature_set '{name}': {missing}")
        cols = [c for c in cols if c in master_df.columns]
    if not cols:
        print(f"[WARN] No usable columns for feature_set '{name}', falling back to BASE_FEATURE_COLUMNS.")
        cols = [c for c in BASE_FEATURE_COLUMNS if c in master_df.columns]
    return _dedupe(cols)


def _resolve_benchmark_dir():
    env_root = os.environ.get('BENCHMARK_RESULTS_ROOT')
    if env_root:
        root = Path(env_root)
    else:
        root = Path('../results/benchmarking')

    # If the env var already points at a classification folder, use it directly
    if root.name.startswith('classification'):
        return root

    meta_dir = root / 'classification'
    if meta_dir.exists():
        return meta_dir
    
    return root

def _normalize_models(model):
    if model is None:
        return ['LSTM', 'BiLSTM', 'GRU', 'BiGRU']
    if isinstance(model, (list, tuple, set)):
        return [str(m) for m in model]
    m = str(model).strip()
    if m.upper() in {'ALL', 'ANY', '*'}:
        return ['LSTM', 'BiLSTM', 'GRU', 'BiGRU']
    return [m]


def select_best_config(model='ALL', horizon=5, metric='mcc', base_dir=None):
    base_dir = Path(base_dir) if base_dir else _resolve_benchmark_dir()
    rows = []
    models = _normalize_models(model)
    if not base_dir.exists():
        raise FileNotFoundError(f'Benchmarking results not found at {base_dir}')

    for group_dir in sorted(base_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        for m in models:
            res_path = group_dir / f"{m}_{horizon}H_results.csv"
            params_path = group_dir / f"{m}_{horizon}H_params.csv"
            if not res_path.exists() or not params_path.exists():
                continue
            df = pd.read_csv(res_path)
            if metric not in df.columns:
                continue
            series = pd.to_numeric(df[metric], errors='coerce')
            if series.dropna().empty:
                continue
            score = float(series.mean())
            params = pd.read_csv(params_path).iloc[0].to_dict()
            feature_set = params.get('params_feature_set', group_dir.name)
            rows.append({
                'model': m,
                'feature_group': group_dir.name,
                'feature_set': FEATURE_SET_ALIAS.get(str(feature_set), str(feature_set)),
                'score': score,
                'params': params,
            })

    if not rows:
        raise FileNotFoundError(f'No {horizon}H benchmarking results found for model={model}')

    rows = sorted(rows, key=lambda r: r['score'], reverse=True)
    best = rows[0]
    print(f"[BEST {horizon}H] model={best['model']} group={best['feature_group']} feature_set={best['feature_set']} {metric}_mean={best['score']:.4f}")
    return best


def build_model_cfg(best_cfg, horizon):
    p = best_cfg['params']
    return {
        'model_type': best_cfg.get('model', p.get('params_model_type', 'LSTM')),
        'feature_set': best_cfg['feature_set'],
        'feature_group': best_cfg['feature_group'],
        'mcc_mean': best_cfg['score'],
        'sequence_length': int(p.get('params_sequence_length', 12)),
        'hidden1': int(p.get('params_hidden1', 128)),
        'hidden2': int(p.get('params_hidden2', 64)),
        'num_layers': int(p.get('params_num_layers', 2)),
        'dropout': float(p.get('params_dropout', 0.3)),
        'inter_rnn_drop': float(p.get('params_inter_rnn_drop', 0.1)),
        'learning_rate': float(p.get('params_learning_rate', 1e-3)),
        'weight_decay': float(p.get('params_weight_decay', 1e-4)),
        'batch_size': float(p.get('params_batch_size', 16)),
        'max_epochs': int(p.get('params_max_epochs', 100)),
        'early_patience': int(p.get('params_early_stopping_patience', 15)),
        'early_min_delta': float(p.get('params_early_stopping_min_delta', 0.0)),
        'lr_patience': int(p.get('params_lr_patience', 7)),
        'lr_factor': float(p.get('params_lr_factor', 0.5)),
        'horizon_steps': int(p.get('params_horizon_steps', horizon)),
    }


def select_and_apply_best_config(horizon=5, model='ALL', metric='mcc'):
    best_cfg = select_best_config(model=model, horizon=horizon, metric=metric, base_dir=Path(os.environ.get('BENCHMARK_RESULTS_ROOT', '/mnt/primary/benchmarking/meta-results')) / 'classification')
    feature_set_name = best_cfg['feature_set']
    feature_cols = _select_feature_cols(feature_set_name)
    cfg = build_model_cfg(best_cfg, horizon)
    print(f"Using model={cfg.get('model_type')} feature_set={feature_set_name} with {len(feature_cols)} columns")
    return cfg, feature_cols, best_cfg


# In[34]:


master_df.reset_index(drop=True, inplace=True)
master_df


# In[35]:


n_train_days = master_df['date'].nunique() - 100
print(f"Using initial training period of {n_train_days} days")


# In[36]:


from pathlib import Path
import json

initial_capital = 100_000.0
H_LIST = [1, 5]
if 'MODEL_CFG' in globals():
    H = int(MODEL_CFG.get('horizon_steps', H))
allow_overlap = True
seed = 42

output_dir = OUTPUT_ROOT / 'simulation' / 'binary' / 'meta'
output_dir.mkdir(parents=True, exist_ok=True)

def _to_scalar(value):
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
    except Exception:
        pass
    return value

def _collect_hyperparams():
    return {
        'H': H,
        'initial_capital': initial_capital,
        'initial_training_days': INITIAL_TRAINING_DAYS,
        'allow_overlap': allow_overlap,
        'early_exit_enabled': early_exit_enabled,
        'seed': seed,
        'allocation_mode': ALLOCATION_MODE,
        'model_save_path': MODEL_SAVE_PATH,
        'min_sequence_length': MIN_SEQUENCE_LENGTH,
        'max_sequence_length': MAX_SEQUENCE_LENGTH,
        'kelly_fraction': KELLY_FRACTION,
        'sector_confidence_threshold': SECTOR_CONFIDENCE_THRESHOLD,
        'retrain_interval': RETRAIN_INTERVAL,
        'max_day_gap': MAX_DAY_GAP,
        'mpt_window_days': MPT_WINDOW_DAYS,
        'mpt_lambda': MPT_LAMBDA,
        'mpt_ridge_eps': MPT_RIDGE_EPS,
        'mpt_max_weight': MPT_MAX_WEIGHT,
        'mpt_long_only': MPT_LONG_ONLY,
        'mpt_use_equal_fallback': MPT_USE_EQUAL_FALLBACK,
        'mpt_only_budget_fraction': MPT_ONLY_BUDGET_FRACTION,
        'max_total_utilization': MAX_TOTAL_UTILIZATION,
        'max_new_daily_utilization': MAX_NEW_DAILY_UTILIZATION,
        'max_ticker_utilization': MAX_TICKER_UTILIZATION,
        'max_sector_utilization': MAX_SECTOR_UTILIZATION,
        'max_position_size': MAX_POSITION_SIZE,
        'max_positions_per_ticker': MAX_POSITIONS_PER_TICKER,
        'min_trade_dollars': MIN_TRADE_DOLLARS,
        'kelly_trade_cap': KELLY_TRADE_CAP,
        'drawdown_throttle_levels': DRAWDOWN_THROTTLE_LEVELS,
        'model_feature_set': MODEL_CFG.get('feature_set') if 'MODEL_CFG' in globals() else None,
        'model_feature_group': MODEL_CFG.get('feature_group') if 'MODEL_CFG' in globals() else None,
        'model_mcc_mean': MODEL_CFG.get('mcc_mean') if 'MODEL_CFG' in globals() else None,
        'model_sequence_length': MODEL_CFG.get('sequence_length') if 'MODEL_CFG' in globals() else None,
        'model_hidden1': MODEL_CFG.get('hidden1') if 'MODEL_CFG' in globals() else None,
        'model_hidden2': MODEL_CFG.get('hidden2') if 'MODEL_CFG' in globals() else None,
        'model_num_layers': MODEL_CFG.get('num_layers') if 'MODEL_CFG' in globals() else None,
        'model_dropout': MODEL_CFG.get('dropout') if 'MODEL_CFG' in globals() else None,
        'model_inter_rnn_drop': MODEL_CFG.get('inter_rnn_drop') if 'MODEL_CFG' in globals() else None,
        'model_learning_rate': MODEL_CFG.get('learning_rate') if 'MODEL_CFG' in globals() else None,
        'model_weight_decay': MODEL_CFG.get('weight_decay') if 'MODEL_CFG' in globals() else None,
        'model_batch_size': MODEL_CFG.get('batch_size') if 'MODEL_CFG' in globals() else None,
        'model_max_epochs': MODEL_CFG.get('max_epochs') if 'MODEL_CFG' in globals() else None,
        'model_early_patience': MODEL_CFG.get('early_patience') if 'MODEL_CFG' in globals() else None,
        'model_early_min_delta': MODEL_CFG.get('early_min_delta') if 'MODEL_CFG' in globals() else None,
        'model_lr_patience': MODEL_CFG.get('lr_patience') if 'MODEL_CFG' in globals() else None,
        'model_lr_factor': MODEL_CFG.get('lr_factor') if 'MODEL_CFG' in globals() else None,
    }

master_df_base = master_df.copy()

for H in H_LIST:
    # select best config per horizon and refresh feature columns
    MODEL_CFG, feature_columns, _best = select_and_apply_best_config(horizon=H, model='ALL', metric='mcc')
    H = int(MODEL_CFG.get('horizon_steps', H))
    master_df = master_df_base.dropna(subset=feature_columns).reset_index(drop=True)
    print(f"After feature-set dropna, master_df shape: {master_df.shape}")

    for early_exit_enabled in [True, False]:
        ee_tag = 'EE_ON' if early_exit_enabled else 'EE_OFF'
        for mode in ['KELLY_ONLY', 'HYBRID_KELLY_MPT', 'MPT_ONLY']:
            ALLOCATION_MODE = mode
            set_global_seeds(seed)
            print('=' * 80)
            print(f'Running allocation mode: {mode}')
            simulation_results = run_simulation(
                master_df,
                feature_columns,
                initial_capital,
                initial_training_days=INITIAL_TRAINING_DAYS,
                H=H,
                allow_overlap=allow_overlap,
                early_exit_enabled=early_exit_enabled
            )
            calculate_final_results(simulation_results, initial_capital)

            hyperparams = _collect_hyperparams()
            if simulation_results.empty:
                results_with_params = pd.DataFrame([{k: _to_scalar(v) for k, v in hyperparams.items()}])
            else:
                results_with_params = simulation_results.copy()
                for key, value in hyperparams.items():
                    results_with_params[key] = _to_scalar(value)

            output_path = output_dir / f'{mode}_H{H}_{ee_tag}.csv'
            results_with_params.to_csv(output_path, index=False)
            print(f'Saved results to {output_path}')


# In[ ]:


# import numpy as np
# import pandas as pd

# all_trades = []
# sector_lookup = None
# if 'sector' in master_df.columns:
#     sector_lookup = master_df.set_index(['ticker'])['sector'].to_dict()

# for _, row in simulation_results.iterrows():
#     exits = row.get('exits_today', [])
#     if not isinstance(exits, list):
#         continue
#     for ex in exits:
#         if not isinstance(ex, dict):
#             continue
#         ticker = ex.get('ticker')
#         if ticker is None:
#             continue
#         allocated = ex.get('allocated_amount', ex.get('invest', np.nan))
#         sector = ex.get('sector')
#         if sector is None and sector_lookup is not None:
#             sector = sector_lookup.get(ticker, 'UNKNOWN')
#         prob_up = ex.get('prob_up', np.nan)
#         expected_rise = ex.get('expected_rise', np.nan)
#         exit_date = ex.get('exit_date')
#         actual_return = ex.get('actual_return', np.nan)
#         pnl = ex.get('pnl', np.nan)
#         exit_reason = ex.get('exit_reason', 'unknown')
#         all_trades.append({
#             'date': ex.get('date', row.get('date')),
#             'exit_date': exit_date,
#             'ticker': ticker,
#             'sector': sector if sector is not None else 'UNKNOWN',
#             'allocated_amount': allocated,
#             'prob_up': prob_up,
#             'expected_rise': expected_rise,
#             'actual_return': actual_return,
#             'pnl': pnl,
#             'exit_reason': exit_reason,
#         })

# trades_df = pd.DataFrame(all_trades)
# print("="*80)
# print("INVESTMENT SUMMARY")
# print("="*80)
# print(f"Total trades: {len(trades_df)}")
# if trades_df.empty:
#     print("No trades recorded.")
# else:
#     print(f"Total capital allocated: ${trades_df['allocated_amount'].sum():,.2f}")
#     print(f"Average position size:  ${trades_df['allocated_amount'].mean():,.2f}")
#     completed = trades_df[trades_df['pnl'].notna()].copy()
#     print(f"Completed trades with returns: {len(completed)}")
#     if not completed.empty:
#         wins = (completed['pnl'] > 0).sum()
#         losses = (completed['pnl'] < 0).sum()
#         print(f"Winning trades: {wins} ({wins/len(completed)*100:.1f}%)")
#         print(f"Losing trades:  {losses} ({losses/len(completed)*100:.1f}%)")
#         total_pnl = completed['pnl'].sum()
#         avg_pnl = completed['pnl'].mean()
#         avg_ret = completed['actual_return'].mean()
#         print(f"Total PnL: ${total_pnl:,.2f}")
#         print(f"Average PnL per trade: ${avg_pnl:,.2f}")
#         print(f"Average return: {avg_ret*100:.2f}%")
#     else:
#         total_pnl = np.nan
#         avg_pnl = np.nan
#         avg_ret = np.nan

#     if 'sector' in trades_df.columns:
#         sector_stats = trades_df.groupby('sector').agg(
#             trades=('ticker','count'),
#             allocated=('allocated_amount','sum'),
#             avg_alloc=('allocated_amount','mean'),
#             pnl_sum=('pnl','sum'),
#             pnl_mean=('pnl','mean'),
#             ret_mean=('actual_return','mean'),
#         ).sort_values('pnl_sum', ascending=False)
#         print("Sector breakdown (sorted by total PnL):")
#         display(sector_stats)

#     if not trades_df.empty:
#         print("TOP 10 BIGGEST EARNERS (trades)")
#         display(trades_df.dropna(subset=['pnl']).sort_values('pnl', ascending=False).head(10))
#         print("TOP 10 BIGGEST LOSERS (trades)")
#         display(trades_df.dropna(subset=['pnl']).sort_values('pnl', ascending=True).head(10))

#     if not trades_df.empty and 'sector' in trades_df.columns:
#         top_sectors = trades_df.dropna(subset=['pnl']).groupby('sector')['pnl'].sum().sort_values(ascending=False)
#         print("TOP 10 SECTORS BY TOTAL PnL")
#         display(top_sectors.head(10).to_frame('total_pnl'))
#         print("BOTTOM 10 SECTORS BY TOTAL PnL")
#         display(top_sectors.tail(10).to_frame('total_pnl'))

#         most_invested_tickers = trades_df['ticker'].value_counts().head(10)
#         print("Top 10 Most Invested Tickers:")
#         display(most_invested_tickers)

#         all_tickers = master_df['ticker'].unique()
#         invested_tickers = trades_df['ticker'].unique()
#         never_invested_tickers = set(all_tickers) - set(invested_tickers)
#         print(f"Tickers Never Invested In ({len(never_invested_tickers)}):")
#         print(never_invested_tickers)


# In[ ]:


folder_path = 'trained_models_meta/'
if os.path.exists(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
else:
    print(f"Folder not found: {folder_path}")

print(f"Contents of {folder_path} after deletion attempt:")
if os.path.exists(folder_path):
    print(os.listdir(folder_path))
else:
    print("Folder does not exist.")

