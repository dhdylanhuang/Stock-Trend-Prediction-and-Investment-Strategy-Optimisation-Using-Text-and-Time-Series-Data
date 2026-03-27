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
from pathlib import Path
import json

warnings.filterwarnings('ignore')


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
MODEL_SAVE_PATH = Path(os.environ.get('MODEL_SAVE_PATH', RUN_ROOT / 'trained_models_multiclass_meta'))
OUTPUT_ROOT = Path(os.environ.get('OUTPUT_ROOT', RUN_ROOT / 'results'))
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print('MASTER_PATH:', MASTER_PATH)
print('EARLY_EXIT_PATH:', EARLY_EXIT_PATH)
print('MODEL_SAVE_PATH:', MODEL_SAVE_PATH)
print('OUTPUT_ROOT:', OUTPUT_ROOT)


# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)


# In[4]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as Fnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext


# ### Configurations

# In[ ]:


MODEL_SAVE_PATH = MODEL_SAVE_PATH if "MODEL_SAVE_PATH" in globals() else Path("trained_models_multiclass_meta")
MIN_SEQUENCE_LENGTH = 12  # Minimum sequence length for any company
MAX_SEQUENCE_LENGTH = 12  # Maximum sequence length to cap computational cost
INITIAL_TRAINING_DAYS = 1109  # Number of days to use for initial training only
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


BUCKET_BINS   = [-np.inf, -0.02, -0.005, 0.0, 0.005, 0.02, np.inf]
K = len(BUCKET_BINS) - 1


# In[8]:


import random


def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seeds(42)


# In[9]:


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


# In[10]:


CALIBRATION_LOGGED = set()


# In[11]:


def bucketize_with_edges(x, edges):
    # x: 1D np array of returns; edges: len K+1
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges)-2).astype(np.int64)

def make_cumulative_targets(y_idx, K):
    # y_idx in [0..K-1] -> T: [N,K-1] with T[:,k]=1(y>k)
    y = y_idx.reshape(-1, 1)
    ks = np.arange(K-1).reshape(1, -1)
    return (y > ks).astype(np.float32)

def ordinal_to_class_probs(P_rep):
    # P_rep: [N,K-1] with P(y>k) (after sigmoid), repaired monotone
    N, K1 = P_rep.shape; K = K1+1
    Pc = np.empty((N, K), dtype=np.float32)
    Pc[:,0] = 1.0 - P_rep[:,0]
    for c in range(1, K-1):
        Pc[:,c] = np.clip(P_rep[:,c-1] - P_rep[:,c], 0., 1.)
    Pc[:,K-1] = P_rep[:,K1-1]
    s = Pc.sum(axis=1, keepdims=True)
    return Pc / np.maximum(s, 1e-8)

def monotone_repair_numpy(P):
    P = np.asarray(P, dtype=np.float32).copy()
    for k in range(P.shape[1]-2, -1, -1):
        P[:,k] = np.maximum(P[:,k], P[:,k+1])
    return P


# In[ ]:


# multiclass calibration tuned for macro-F1 balance
def _class_dist(y_hat, K):
    hist = np.bincount(y_hat, minlength=K).astype(np.float32)
    return hist / max(1, y_hat.size)

def _balance_penalty(p_hat, p_target, norm="l1"):
    if norm == "l1":
        return float(np.abs(p_hat - p_target).sum())
    elif norm == "l2":
        return float(np.sqrt(((p_hat - p_target) ** 2).sum()))
    else:
        return float(np.abs(p_hat - p_target).sum())

def decode_ordinal_with_taus(P_rep, taus=None, taus_override=None):
    # decode ordinal cumulative probs to class indices using per-head taus.
    N, K_1 = P_rep.shape
    if taus is None:
        taus = np.full(K_1, 0.5, dtype=np.float32)
    if taus_override:
        taus = taus.copy()
        for k, v in taus_override.items():
            taus[k] = v
    comp = (P_rep >= taus.reshape(1, -1)).astype(np.int32)
    return comp.sum(axis=1).astype(np.int64)

def _macro_f1_torch(y_hat_batch, y_true_oh, K, device):
    cls_range = torch.arange(K, device=device)
    pred_oh = (y_hat_batch.unsqueeze(-1) == cls_range)
    true_oh_b = y_true_oh.unsqueeze(0).bool()
    tp = (pred_oh & true_oh_b).float().sum(dim=1)
    fp = (pred_oh & ~true_oh_b).float().sum(dim=1)
    fn = (~pred_oh & true_oh_b).float().sum(dim=1)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
    present = y_true_oh.any(dim=0)
    f1 = f1 * present.float()
    return f1.sum(dim=1) / present.float().sum().clamp(min=1.0)

def tune_temperature_and_taus(
    Z_val,
    y_val_idx,
    K,
    T_grid=np.linspace(0.6, 2.5, 15),
    tau_grid=np.linspace(0.2, 0.8, 31),
    lambda_balance=0.05,
    target_priors="uniform",
    coord_rounds=3,
):
    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps') if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    Z_val_t = (Z_val.detach().to(device).float()
               if isinstance(Z_val, torch.Tensor)
               else torch.tensor(Z_val, dtype=torch.float32, device=device))

    N, K_1 = Z_val_t.shape
    assert K_1 == K - 1

    y_t = torch.tensor(np.asarray(y_val_idx), dtype=torch.long, device=device)
    y_oh = torch.zeros(N, K, dtype=torch.float32, device=device)
    y_oh.scatter_(1, y_t.unsqueeze(1), 1.0)

    if isinstance(target_priors, str) and target_priors == "uniform":
        p_target = torch.full((K,), 1.0 / K, dtype=torch.float32, device=device)
    else:
        pt = torch.tensor(np.asarray(target_priors), dtype=torch.float32, device=device)
        p_target = pt / pt.sum()

    T_grid_t = torch.tensor(np.asarray(T_grid), dtype=torch.float32, device=device)
    tau_grid_t = torch.tensor(np.asarray(tau_grid), dtype=torch.float32, device=device)
    n_tau = len(tau_grid_t)
    cls_range = torch.arange(K, device=device)

    def _score_batch(y_hat_batch):
        f1s = _macro_f1_torch(y_hat_batch, y_oh, K, device)
        p_hat = (y_hat_batch.unsqueeze(-1) == cls_range).float().mean(dim=1)
        pen = (p_hat - p_target).abs().sum(dim=-1)
        return f1s - lambda_balance * pen

    best_score = -1e9
    best_T = float(T_grid_t[0])
    best_taus = torch.full((K_1,), 0.5, dtype=torch.float32, device=device)

    for T in T_grid_t:
        P_rep, _ = torch.cummin(torch.sigmoid(Z_val_t / T), dim=1)

        taus = torch.full((K_1,), 0.5, dtype=torch.float32, device=device)

        for _ in range(coord_rounds):
            for k in range(K_1):
                taus_batch = taus.unsqueeze(0).expand(n_tau, -1).clone()
                taus_batch[:, k] = tau_grid_t

                comp = (P_rep.unsqueeze(0) >= taus_batch.unsqueeze(1)).int()
                y_hat_b = comp.sum(dim=-1)

                taus[k] = tau_grid_t[_score_batch(y_hat_b).argmax()]

        y_hat_f = (P_rep >= taus).int().sum(dim=-1).unsqueeze(0)
        score = float(_score_batch(y_hat_f).item())
        if score > best_score:
            best_score = score
            best_T = float(T.item())
            best_taus = taus.clone()

    return best_T, best_taus.cpu().numpy().astype(np.float32)


# #### Define Horizon Target
# For a horizon H, compute both the direction and H-day return off the same base day t

# In[13]:


def add_horizon_targets(df: pd.DataFrame, H: int, price_col='close') -> pd.DataFrame:
    df = df.sort_values(['ticker','date']).copy()
    df[f'ret_{H}d']    = df.groupby('ticker')[price_col].shift(-H) / df[price_col] - 1.0
    df[f'target_{H}d'] = (df[f'ret_{H}d'] > 0).astype(int)
    return df


# #### Identify Contiguous Periods
# Groups rows for a ticker into blocks where data gaps never exceed MAX_DAY_GAP

# In[14]:


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
            # end current period and start new one
            contiguous_periods.append((start_idx, i-1))
            start_idx = i


    contiguous_periods.append((start_idx, len(dates)-1))


    contiguous_periods = [(s, e) for s, e in contiguous_periods if e - s >= MIN_SEQUENCE_LENGTH]

    return contiguous_periods


# #### Calculate Dynamic Sequence Length 
# Pick a sequence length tailored to the amount of history available for a ticker.
# Avoids hard-coding a single window size for short vs long histories.

# In[15]:


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

# In[16]:


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

# In[17]:


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

# In[18]:


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

# In[19]:


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

# In[20]:


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

# In[21]:


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

# In[22]:


class OrdinalSequenceDataset(Dataset):
    def __init__(self, X, T):  # T: float32 matrix [N, K-1] with 0/1
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.T[i]
    
class RNNOrdinal(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, fc=32,
                 dropout=0.3, inter_dropout=0.1, head='CORAL', K=6,
                 model_type='LSTM', num_layers=2):
        super().__init__()
        self.K = K
        self.model_type = model_type
        self.num_layers = int(num_layers)
        if self.num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")

        bidirectional = model_type in ('BiLSTM', 'BiGRU')
        rnn_cls = nn.LSTM if model_type in ('LSTM', 'BiLSTM') else nn.GRU

        self.rnn1 = rnn_cls(
            n_features, hidden1, num_layers=1, batch_first=True,
            bidirectional=bidirectional, dropout=0.1
        )
        self.inter_drop = nn.Dropout(inter_dropout)
        self.rnn2 = None

        if self.num_layers == 2:
            in2 = hidden1 * (2 if bidirectional else 1)
            self.rnn2 = rnn_cls(
                in2, hidden2, num_layers=1, batch_first=True,
                bidirectional=bidirectional, dropout=0.1
            )
            feat_dim = hidden2 * (2 if bidirectional else 1)
        else:
            feat_dim = hidden1 * (2 if bidirectional else 1)

        self.norm = nn.BatchNorm1d(feat_dim)
        self.fc1 = nn.Linear(feat_dim, fc)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        if head.upper() == 'CORAL':
            self.w = nn.Linear(fc, 1, bias=False)
            self._beta = nn.Parameter(torch.zeros(K-1))
            self.softplus = nn.Softplus()
            self.head_type = 'CORAL'
        else:
            self.fc_out = nn.Linear(fc, K-1)
            self.head_type = 'CORN'

    def forward(self, x):
        out1, _ = self.rnn1(x)
        if self.num_layers == 2:
            out1 = self.inter_drop(out1)
            out2, _ = self.rnn2(out1)
            last = out2[:, -1, :]
        else:
            last = out1[:, -1, :]

        last = self.norm(last)
        z = self.relu(self.fc1(last))
        z = self.drop(z)
        if self.head_type == 'CORAL':
            base = self.w(z)                              # [B,1]
            deltas = self.softplus(self._beta)            # >=0
            b = torch.cumsum(deltas, dim=0)               # [K-1]
            logits = base - b                             # broadcast -> [B,K-1]
        else:
            logits = self.fc_out(z)                       # [B,K-1]
        return logits

def _ord_model_key(model_type: str) -> str:
    return str(model_type or 'LSTM').strip().lower()


def _ord_state_name(ticker: str, H: int, model_type: str) -> str:
    key = _ord_model_key(model_type)
    if key == 'lstm':
        return _ord_artifact_name(ticker, H, model_type, "state")
    return f"{ticker}_{key}ORD_H{H}.pt"


def _ord_artifact_name(ticker: str, H: int, model_type: str, kind: str) -> str:
    key = _ord_model_key(model_type)
    if kind == 'state':
        return _ord_state_name(ticker, H, model_type)

    if key == 'lstm':
        if kind == 'scaler':
            return _ord_artifact_name(ticker, H, model_type, "scaler")
        if kind == 'seq':
            return _ord_artifact_name(ticker, H, model_type, "seq")
        if kind == 'edges':
            return _ord_artifact_name(ticker, H, model_type, "edges")
        if kind == 'mu_c':
            return _ord_artifact_name(ticker, H, model_type, "mu_c")
        if kind == 'taus':
            return _ord_artifact_name(ticker, H, model_type, "taus")
        if kind == 'temp':
            return _ord_artifact_name(ticker, H, model_type, "temp")

    if kind == 'scaler':
        return f"{ticker}_scaler_{key}_H{H}.pkl"
    if kind == 'seq':
        return f"{ticker}_seq_length_{key}_H{H}.pkl"
    if kind == 'edges':
        return f"{ticker}_edges_{key}_H{H}.pkl"
    if kind == 'mu_c':
        return f"{ticker}_mu_c_{key}_H{H}.pkl"
    if kind == 'taus':
        return f"{ticker}_taus_{key}_H{H}.pkl"
    if kind == 'temp':
        return f"{ticker}_temperature_{key}_H{H}.pkl"
    raise ValueError(f"Unknown ordinal artifact kind: {kind}")

class EarlyStopper:
    def __init__(self, patience=15, mode='min', min_delta=0.0):
        self.patience = patience
        self.min_delta = float(min_delta)
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

# In[ ]:


def train_company_models(company_data_df: pd.DataFrame,
                                 ticker: str,
                                 feature_cols: list,
                                 model_save_path: str,
                                 sequence_length: int = None,
                                 H: int = 1,
                                 K: int = K,        # global K
                                 head_type: str = 'CORAL',
                                 cfg: dict | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mixed_precision = (device.type == 'cuda' and USE_MIXED_PRECISION)
    amp_scaler = torch.amp.GradScaler('cuda', enabled=mixed_precision)

    if cfg is None:
        cfg = globals().get('MODEL_CFG', {})

    print(f"[TRAIN] {ticker}: preparing data (H={H}) with {len(company_data_df)} rows")

    ret_col = f"ret_{H}d"
    if ret_col not in company_data_df.columns:
        company_data_df = add_horizon_targets(company_data_df, H=H, price_col='close')

    if sequence_length is None:
        sequence_length = int(cfg.get('sequence_length', 0)) or calculate_dynamic_sequence_length(company_data_df)
    if len(company_data_df) < sequence_length + H + 10:
        print(f"[TRAIN] {ticker}: insufficient history for sequence length {sequence_length}")
        return False, sequence_length

    contigs = identify_contiguous_periods(company_data_df)
    if not contigs:
        print(f"[TRAIN] {ticker}: no contiguous periods found")
        return False, sequence_length

    rets = company_data_df[ret_col].values
    y_idx_all = bucketize_with_edges(rets, BUCKET_BINS)
    X_all = company_data_df[feature_cols].values

    X_seq, y_seq = create_contiguous_sequences(
        X_all, y_idx_all, contigs, sequence_length, H
    )
    if len(X_seq) < 40:
        print(f"[TRAIN] {ticker}: too few sequences ({len(X_seq)})")
        return False, sequence_length
    print(f"[TRAIN] {ticker}: built {len(X_seq)} sequences (seq_len={sequence_length})")

    X_tr, X_val, y_tr, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    scaler = MinMaxScaler()
    F = X_tr.shape[-1]
    X_tr = scaler.fit_transform(X_tr.reshape(-1, F)).reshape(X_tr.shape)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    print(f"[TRAIN] {ticker}: scaler fitted on {X_tr.shape[0]} training sequences")
    # sanity checks
    assert F == len(feature_cols),         f"[TRAIN] {ticker}: feature dimension mismatch: X_tr has {F}, feature_cols has {len(feature_cols)}"

    print(f"[TRAIN] {ticker}: scaler.n_features_in_ = {scaler.n_features_in_}, len(feature_cols) = {len(feature_cols)}")
    assert scaler.n_features_in_ == len(feature_cols),         f"[TRAIN] {ticker}: scaler expects {scaler.n_features_in_} features, but feature_cols has {len(feature_cols)}"

    # Config-driven hyperparams
    model_type = str(cfg.get('model_type', 'LSTM'))
    num_layers = int(cfg.get('num_layers', 2))
    hidden1 = int(cfg.get('hidden1', 128))
    hidden2 = int(cfg.get('hidden2', 64))
    dropout = float(cfg.get('dropout', 0.3))
    inter_dropout = float(cfg.get('inter_rnn_drop', 0.1))
    head_type = str(cfg.get('ordinal_head', head_type))

    K_cfg = int(cfg.get('n_classes', K))
    if K_cfg != K:
        print(f"[WARN] {ticker}: cfg n_classes={K_cfg} does not match BUCKET_BINS K={K}; using K={K}")
    K_used = K

    T_tr  = make_cumulative_targets(y_tr, K_used)
    T_val = make_cumulative_targets(y_val, K_used)

    train_ds = OrdinalSequenceDataset(X_tr, T_tr)
    val_ds   = OrdinalSequenceDataset(X_val, T_val)
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

    n_features = X_tr.shape[-1]
    model = RNNOrdinal(
        n_features=n_features,
        hidden1=hidden1,
        hidden2=hidden2,
        dropout=dropout,
        inter_dropout=inter_dropout,
        head=head_type,
        K=K_used,
        model_type=model_type,
        num_layers=num_layers,
    ).to(device)

    learning_rate = float(cfg.get('learning_rate', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    pos_rate = T_tr.mean(axis=0)
    pos_w = ((1.0 - pos_rate) / np.clip(pos_rate, 1e-6, 1.0)).astype(np.float32)
    pos_w = torch.tensor(pos_w, dtype=torch.float32, device=device)
    print(f"[TRAIN] {ticker}: model n_features = {n_features}")

    def loss_fn(logits, T):
        losses = []
        for k in range(logits.shape[1]):
            losses.append(Fnn.binary_cross_entropy_with_logits(logits[:,k], T[:,k], pos_weight=pos_w[k]))
        return torch.stack(losses).mean()

    early_patience = int(cfg.get('early_stopping_patience', 15))
    early_min_delta = float(cfg.get('early_stopping_min_delta', 0.0))
    early = EarlyStopper(patience=early_patience, mode='min', min_delta=early_min_delta)

    lr_factor = float(cfg.get('lr_factor', 0.5))
    lr_patience = int(cfg.get('lr_patience', 7))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=1e-7)

    max_epochs = int(cfg.get('max_epochs', 100))
    for epoch in range(max_epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.amp.autocast('cuda') if mixed_precision else nullcontext()
            with ctx:
                logits = model(xb)
                l = loss_fn(logits, yb)
            if mixed_precision:
                amp_scaler.scale(l).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                l.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot += l.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            vtot = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                l = loss_fn(logits, yb)
                vtot += l.item() * xb.size(0)
            val_loss = vtot / len(val_loader.dataset)

        scheduler.step(val_loss)
        _ = early.step(val_loss, model)
        if early.counter >= early.patience:
            break

    if early.best_state_dict is not None:
        model.load_state_dict(early.best_state_dict)

    ret_seq_full = company_data_df[ret_col].values[sequence_length:]
    ret_train = ret_seq_full[:len(y_tr)]
    mu_c = np.array([ret_train[y_tr==c].mean() if np.any(y_tr==c) else 0.0 for c in range(K_used)], dtype=np.float32)

    taus = None
    temperature = 1.0
    try:
        model.eval()
        batch = 256
        logits_list = []
        for start in range(0, len(X_val), batch):
            xb = torch.from_numpy(X_val[start:start+batch].astype(np.float32)).to(device)
            with torch.no_grad():
                logits = model(xb)
            logits_list.append(logits.detach())
        if logits_list:
            val_logits = torch.cat(logits_list, dim=0)
            t_min = float(cfg.get('t_min', 0.6))
            t_max = float(cfg.get('t_max', 2.5))
            t_steps = int(cfg.get('t_steps', 15))
            tau_min = float(cfg.get('tau_min', 0.2))
            tau_max = float(cfg.get('tau_max', 0.8))
            tau_steps = int(cfg.get('tau_steps', 31))
            lambda_balance = float(cfg.get('lambda_balance', 0.05))
            coord_rounds = int(cfg.get('coord_rounds', 3))

            T_star, taus = tune_temperature_and_taus(
                Z_val=val_logits,
                y_val_idx=y_val,
                K=K_used,
                T_grid=np.linspace(t_min, t_max, t_steps),
                tau_grid=np.linspace(tau_min, tau_max, tau_steps),
                lambda_balance=lambda_balance,
                target_priors="uniform",
                coord_rounds=coord_rounds,
            )
            temperature = float(T_star)
            print(f"[CAL] {ticker}: taus={np.round(taus, 3)} | temperature={temperature:.3f}")
        else:
            print(f"[CAL] {ticker}: no validation logits -> skipping calibration")
    except Exception as exc:
        print(f"[CAL] {ticker}: calibration failed ({exc})")
        taus = None
        temperature = 1.0

    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "state")))
    joblib.dump(scaler, os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "scaler")))
    joblib.dump(sequence_length, os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "seq")))
    joblib.dump(BUCKET_BINS, os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "edges")))
    joblib.dump(mu_c, os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "mu_c")))
    if taus is not None and len(taus) == (K_used - 1) and np.all(np.isfinite(taus)):
        joblib.dump(taus, os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "taus")))
        print(f"[SAVE] {ticker}: stored taus")
    if temperature is not None and np.isfinite(temperature):
        joblib.dump(float(temperature), os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "temp")))
        print(f"[SAVE] {ticker}: stored temperature scale {temperature:.3f}")
    print(f"[SAVE] {ticker}: artifacts persisted (H={H})")

    del model, scaler, train_loader, val_loader, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return True, sequence_length


# #### Predict Next Day Performance 
# One-step-ahead inference for a ticker on a given day.
# 
# Feeds calibrated p(up) into the portfolio/Kelly step.

# In[24]:


def predict_next_horizon(company_data_df: pd.DataFrame,
                         ticker: str,
                         feature_cols: list,
                         model_save_path: str,
                         H: int,
                         K: int = 6,
                         head: str = "CORAL",
                         cfg: dict | None = None) -> dict | None:
    # Ordinal inference with optional calibration artifacts.
    if cfg is None:
        cfg = globals().get('MODEL_CFG', {})

    model_type = str(cfg.get('model_type', 'LSTM'))

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state_dict_path = os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "state"))
        if not os.path.isfile(state_dict_path):
            return None

        scaler   = joblib.load(os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "scaler")))
        seq_len  = joblib.load(os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "seq")))
        mu_c     = joblib.load(os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "mu_c")))

        taus_path = os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "taus"))
        T_path    = os.path.join(model_save_path, _ord_artifact_name(ticker, H, model_type, "temp"))
        taus = joblib.load(taus_path) if os.path.isfile(taus_path) else None
        temperature = float(joblib.load(T_path)) if os.path.isfile(T_path) else 1.0

        num_layers = int(cfg.get('num_layers', 2))
        hidden1 = int(cfg.get('hidden1', 128))
        hidden2 = int(cfg.get('hidden2', 64))
        dropout = float(cfg.get('dropout', 0.3))
        inter_dropout = float(cfg.get('inter_rnn_drop', 0.1))
        head = str(cfg.get('ordinal_head', head))

        K_cfg = int(cfg.get('n_classes', K))
        if K_cfg != K:
            print(f"[WARN] {ticker}: cfg n_classes={K_cfg} does not match BUCKET_BINS K={K}; using K={K}")
        K_used = K

        n_features = len(feature_cols)
        model = RNNOrdinal(
            n_features=n_features,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
            inter_dropout=inter_dropout,
            head=head.upper(),
            K=K_used,
            model_type=model_type,
            num_layers=num_layers,
        ).to(device)
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        model.eval()
    except Exception:
        return None

    if 'CALIBRATION_LOGGED' not in globals():
        globals()['CALIBRATION_LOGGED'] = set()
    if ticker not in CALIBRATION_LOGGED:
        if taus is not None:
            print(f"[CAL] {ticker}: loaded taus {np.round(taus,3)}")
        else:
            print(f"[CAL] {ticker}: no taus artifact found")
        print(f"[CAL] {ticker}: temperature scaling {temperature:.3f}")
        CALIBRATION_LOGGED.add(ticker)

    contiguous = identify_contiguous_periods(company_data_df)
    if not contiguous:
        return None
    start, end = contiguous[-1]
    period = company_data_df.iloc[start:end+1]
    if len(period) < seq_len:
        return None

    try:
        last_seq = period.tail(seq_len)
        X_scaled = scaler.transform(last_seq[feature_cols])
    except Exception:
        return None

    x = torch.from_numpy(np.asarray([X_scaled], dtype=np.float32)).to(device)

    with torch.no_grad():
        ctx = torch.amp.autocast('cuda') if (torch.cuda.is_available() and USE_MIXED_PRECISION) else nullcontext()
        with ctx:
            z = model(x)
        if temperature and temperature > 0 and temperature != 1.0:
            z = z / temperature
        P_cum = torch.sigmoid(z).cpu().numpy()
        P_cum = monotone_repair_numpy(P_cum)
        Pc    = ordinal_to_class_probs(P_cum)

    mu_c = np.asarray(mu_c, dtype=np.float32).reshape(-1)
    if mu_c.shape[0] != K_used:
        return None
    expected_rise = float((Pc[0] * mu_c).sum())

    top_bucket_idx = K_used - 1
    prob_top_bucket = float(Pc[0, top_bucket_idx])
    bucket_idx = int(np.argmax(Pc[0]))

    class_idx = None
    if taus is not None:
        taus = np.asarray(taus, dtype=np.float32).reshape(-1)
        if taus.shape[0] == (K_used - 1):
            class_idx = int(decode_ordinal_with_taus(P_cum, taus=taus)[0])

    positive_buckets = [i for i in range(K_used) if BUCKET_BINS[i] >= 0]
    prob_up = float(Pc[0, positive_buckets].sum()) if positive_buckets else prob_top_bucket

    return {
        'ticker': ticker,
        'expected_rise': expected_rise,
        'predicted_prob': prob_up,  # probability of positive move
        'prob_up': prob_up,
        'prob_bucket_top': prob_top_bucket,
        'bucket_idx': bucket_idx,
        'class_idx': class_idx,
        'class_probs': Pc[0].tolist(),
    }


# #### Select and Size Portfolio
# Turn a cross-section of calibrated predictions into position sizes.
# 
# Enforces diversification (one per sector) and risk discipline (Fractional Kelly).

# In[25]:


def select_and_size_portfolio(daily_predictions_df: pd.DataFrame, payout_map: dict,
                              total_capital: float, sector_threshold: float,
                              kelly_fraction: float, top_k: int = 3, softmax_alpha: float = 1) -> pd.DataFrame:
    """Pick one name per sector via softmax over top-K expected_rise, then size with Kelly."""
    if daily_predictions_df.empty:
        print("[PORT] No predictions provided today.")
        return pd.DataFrame()

    print(f"[PORT] Received {len(daily_predictions_df)} predictions for sizing.")
    df = daily_predictions_df.copy()

    if 'prob_up' not in df.columns and 'class_probs' in df.columns:
        positive_buckets = [i for i in range(K) if BUCKET_BINS[i] >= 0]
        def _sum_up(probs):
            if isinstance(probs, (list, tuple)) and len(probs) == K and positive_buckets:
                return float(sum(probs[i] for i in positive_buckets))
            return np.nan
        df['prob_up'] = df['class_probs'].apply(_sum_up)

    if 'prob_up' not in df.columns and 'predicted_prob' in df.columns:
        df['prob_up'] = df['predicted_prob']

    if 'expected_rise' not in df.columns:
        print("[PORT] Missing expected_rise; cannot size portfolio.")
        return pd.DataFrame()

    # base filters
    df = df[(df['prob_up'] >= sector_threshold) & (df['expected_rise'] > 0)]
    print(f"[PORT] {len(df)} tickers cleared prob_up ≥ {sector_threshold:.3f} and positive expected return.")
    if df.empty:
        return pd.DataFrame()

    candidates = []
    for _, row in df.iterrows():
        ticker = row['ticker']
        p_up = float(row['prob_up'])
        payout = payout_map.get(ticker, {})
        b = payout.get('b') if isinstance(payout, dict) else float(payout)
        if not np.isfinite(b) or b <= 0:
            continue
        kelly_pct = p_up - (1.0 - p_up) / b
        if kelly_pct <= 0:
            continue
        investment_fraction = min(MAX_POSITION_SIZE, kelly_pct * kelly_fraction)
        investment_fraction = max(0.0, investment_fraction)
        investment_amount = total_capital * investment_fraction
        if investment_amount < MIN_TRADE_DOLLARS:
            continue
        enriched = row.to_dict()
        enriched.update({
            'kelly_pct': kelly_pct,
            'investment_fraction': investment_fraction,
            'investment_amount': investment_amount,
        })
        candidates.append(enriched)

    if not candidates:
        print("[PORT] All candidates rejected after quality filters.")
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates)

    rows = []
    alpha = max(softmax_alpha, 1e-6)
    for sector, group in cand_df.groupby('sector'):
        if group.empty:
            continue
        top_group = group.sort_values('expected_rise', ascending=False).head(max(1, top_k))
        logits = (top_group['expected_rise'].to_numpy() * 100.0 * alpha)
        logits = logits - logits.max()  # stabilize
        weights = np.exp(logits)
        probs = weights / weights.sum()
        choice_idx = np.random.choice(len(top_group), p=probs)
        choice = top_group.iloc[choice_idx]

        print(f"[PORT] Sector {sector}: sampled {choice['ticker']} from top-{len(top_group)} (expected={choice['expected_rise']:.4f}, p_up={choice['prob_up']:.3f})")

        rows.append({
            'ticker': choice['ticker'],
            'sector': sector,
            'prob_up': float(choice['prob_up']),
            'predicted_prob': float(choice['prob_up']),
            'expected_rise': choice['expected_rise'],
            'bucket_idx': choice.get('bucket_idx', np.nan),
            'investment_fraction': choice['investment_fraction'],
            'investment_amount': choice['investment_amount'],
        })

    if not rows:
        print("[PORT] All sector winners rejected after sizing.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# #### Models Exist for Ticker
# Quick guard to avoid retraining if a ticker’s artifacts already exist.
# 
# Decide whether to train.

# In[26]:


def models_exist_for_ticker(ticker: str, model_path: str, H: int = 1, model_type: str = 'LSTM') -> bool:
    """Check the minimal set of artifacts needed for ordinal inference."""
    required = [
        _ord_artifact_name(ticker, H, model_type, 'state'),
        _ord_artifact_name(ticker, H, model_type, 'scaler'),
        _ord_artifact_name(ticker, H, model_type, 'seq'),
        _ord_artifact_name(ticker, H, model_type, 'mu_c'),
    ]
    return all(os.path.exists(os.path.join(model_path, f)) for f in required)


# #### Run Simulation
# The “engine” — walks forward over dates, trains as needed, predicts, sizes, books PnL, and updates capital.
# 
# Enforces chronological integrity, periodic retraining, diversification, and proper capital tracking.

# In[27]:


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


# In[28]:


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

    # precompute realized returns for settlement only (not used for training)
    master_df_returns = add_horizon_targets(master_df.copy(), H=H, price_col='close')
    ret_lookup = master_df_returns.set_index(['ticker', 'date'])[ret_col].to_dict()
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

    # overlapping book
    open_positions = []  # each: {'ticker','invest','entry_date','exit_date',...}
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
                    if VERBOSE:
                        print(f"[EXIT] {current_date}: {pos['ticker']} r={r_exit:.4f}, PnL=${pnl:.2f}")
                    capital += pos['invest'] + pnl  # release capital
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
                if float(p_up_1d) < early_exit_threshold or (int(reg_dir) == -1 and float(reg_conf) > 0.2):
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

        # equity at START (after settlements, before new entries)
        open_notional_start = sum(p['invest'] for p in open_positions)
        equity_start = capital + open_notional_start
        daily_return = (equity_start / prev_equity_end) - 1.0 if prev_equity_end > 0 else 0.0

        # build prediction universe using info strictly before current_date
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
            if need_retrain or not have_models:
                print("Retraining model for ticker:", ticker)
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
        if VERBOSE:
            print(f"[SIM] {current_date}: built {len(daily_predictions_df)} predictions")
        if (not daily_predictions_df.empty and
            'predicted_prob' not in daily_predictions_df.columns and
            'calibrated_prediction' not in daily_predictions_df.columns):
            print("[WARN] predicted_prob missing from daily_predictions_df columns:",
                  sorted(daily_predictions_df.columns))


        # size raw trades via Kelly
        investment_decision_df = select_and_size_portfolio(
            daily_predictions_df,
            payout_map_day,
            capital,
            sector_threshold=SECTOR_CONFIDENCE_THRESHOLD,
            kelly_fraction=KELLY_FRACTION,
        )
        if investment_decision_df.empty:
            if VERBOSE:
                print(f"[SIM] {current_date}: no trades proposed today.")
        else:
            if VERBOSE:
                print(f"[SIM] {current_date}: {len(investment_decision_df)} trades proposed.")

        investment_decision_df, alloc_meta = apply_allocation_mode(
            investment_decision_df,
            historical_data,
            current_date,
            capital,
            open_positions,
            allocation_mode=ALLOCATION_MODE,
        )

        # entry logic (allow_overlap controls laddering)
        exit_idx = i + H
        if exit_idx < len(unique_dates):
            exit_date = pd.to_datetime(unique_dates[exit_idx])
        else:
            exit_date = None  # can't open new trades that we cannot exit within data

        entries_today = []
        if exit_date is not None and not investment_decision_df.empty:
            # ensure the column exists even if future changes rename it
            if 'predicted_prob' not in investment_decision_df.columns:
                if 'calibrated_prediction' in investment_decision_df.columns:
                    investment_decision_df = investment_decision_df.rename(
                        columns={'calibrated_prediction': 'predicted_prob'}
                    )
                elif 'score' in investment_decision_df.columns:
                    investment_decision_df = investment_decision_df.rename(
                        columns={'score': 'predicted_prob'}
                    )
                else:
                    # nothing to sort by
                    investment_decision_df = pd.DataFrame()

            if not investment_decision_df.empty:
                investment_decision_df = investment_decision_df.sort_values('predicted_prob', ascending=False)


            # current exposures (post-settlement)
            util_mult = utilization_throttle(equity_curve)
            total_open, expo_ticker, expo_sector = compute_exposures(open_positions, sectors_by_ticker)
            new_today_open = 0.0
            budget_post_caps = 0.0

            for _, trade in investment_decision_df.iterrows():
                tkr = trade['ticker']
                sec = sectors_by_ticker.get(tkr, 'UNKNOWN')

                # per-ticker ladder depth
                open_count = sum(1 for p in open_positions if p['ticker'] == tkr)
                if allow_overlap:
                    if open_count >= MAX_POSITIONS_PER_TICKER:
                        continue
                else:
                    if open_count >= 1:
                        continue  # no overlapping allowed

                # base size from Kelly (already scaled by KELLY_FRACTION)
                base_amt = float(trade['investment_amount'])
                base_amt = min(base_amt, KELLY_TRADE_CAP * (capital + sum(p['invest'] for p in open_positions)))  # vs equity

                # clamp by remaining budgets
                amt = clamp_by_caps(base_amt, (capital + sum(p['invest'] for p in open_positions)),
                                    total_open, new_today_open, expo_ticker, expo_sector,
                                    tkr, sec, util_mult)
                if amt < MIN_TRADE_DOLLARS:
                    continue

                # lock capital and record position
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
                if VERBOSE:
                    print(f"[TRADE] {current_date}: allocated ${amt:.2f} to {tkr} exiting on {exit_date}")

        # equity at END (after entries)
        open_notional_end = sum(p['invest'] for p in open_positions)
        equity_end = capital + open_notional_end

        # log the day
        simulation_log.append({
            'date': current_date,
            'capital_start': equity_start - open_notional_start,  # cash only
            'capital_end': capital,                               # cash only
            'equity_start': equity_start,
            'equity_end': equity_end,
            'daily_pnl_realized': total_pnl_today,                # realized PnL today
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

# In[30]:


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

# In[31]:


master_df = pd.read_parquet(MASTER_PATH)


# In[32]:


master_df


# In[ ]:


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


# In[ ]:


feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'roll_ret_1d', 'roll_ret_5d', 
    'roll_ret_20d',
    
    'ema_12', 'ema_26', 'ema_50', 'macd_12_26_9', 'macdh_12_26_9',
    'macds_12_26_9', 'rsi_14', 'stochrsik_14_14_3_3', 'stochrsid_14_14_3_3',
    'atrr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'obv',
]

BASE_FEATURE_COLUMNS = feature_columns.copy()


# feature_columns.extend(new_indicator_columns)

all_pipelines = {}
all_results_dfs = {}
all_analyses = {}


# In[ ]:


print(master_df.shape)
master_df = master_df.dropna(subset=feature_columns).sort_values(['ticker','date'])
print(master_df.shape)


# In[ ]:


FEATURE_SET_ALIAS = {
    'sentinment': 'sentiment',
}


# In[ ]:


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


# In[ ]:


# Feature sets
feature_sets = {
    # 'base': feature_columns,
    # 'sentiment': feature_columns + sentiment_columns,,
    # 'emotion': feature_columns + emotion_columns,
    # 'unified_emotion': feature_columns + unified_emotion_columns,
    # 'finbert': feature_columns + finbert_columns,
    # 'all_nlp': feature_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    # 'sector': feature_columns + sector_columns,
    # 'sector_sentiment': feature_columns + sector_columns + sentiment_columns,
    # 'sector_emotion': feature_columns + sector_columns + emotion_columns,
    # 'sector_unified_emotion': feature_columns + sector_columns + unified_emotion_columns,
    # 'sector_finbert': feature_columns + sector_columns + finbert_columns,
    # 'sector_all_nlp': feature_columns + sector_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,
    
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
    'meta_sector_stance': feature_columns + meta_regression_columns + meta_classification_columns + sector_columns + stance_columns,
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
    'meta_err_sector_stance': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + stance_columns,
    'meta_err_sector_emotion': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + emotion_columns,
    'meta_err_sector_unified_emotion': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + unified_emotion_columns,
    'meta_err_sector_finbert': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + finbert_columns,
    'meta_err_sector_all_nlp': feature_columns + meta_regression_columns + meta_classification_columns + meta_regression_err_columns + meta_classification_err_columns + sector_columns + sentiment_columns + emotion_columns + unified_emotion_columns + stance_columns + finbert_columns,    
    
}


# Grouping of feature sets to tune
feature_groups = {
    'base': ['meta', 'meta_err'],
    'nlp': ['meta_sentiment', 'meta_stance', 'meta_emotion', 'meta_unified_emotion', 'meta_finbert', 'meta_all_nlp', 'meta_err_sentiment', 'meta_err_stance', 'meta_err_emotion', 'meta_err_unified_emotion', 'meta_err_finbert', 'meta_err_all_nlp'],
    'sector': ['meta_sector', 'meta_sector_sentiment', 'meta_sector_stance', 'meta_sector_emotion', 'meta_sector_unified_emotion', 'meta_sector_finbert', 'meta_sector_all_nlp', 'meta_err_sector', 'meta_err_sector_sentiment', 'meta_err_sector_stance', 'meta_err_sector_emotion', 'meta_err_sector_unified_emotion', 'meta_err_sector_finbert', 'meta_err_sector_all_nlp'],
}



# In[ ]:


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
    root = Path('/mnt/primary/benchmarking/meta-results')

    # If env_root already points at a multiclass folder, use it directly.
    if root.name.startswith('multiclass'):
        return root

    ta_dir = root / 'multiclass'
    meta_dir = root / 'multiclass'
    if ta_dir.exists():
        return ta_dir
    if meta_dir.exists():
        return meta_dir
    # fallback for error message
    return ta_dir


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
        'n_classes': int(p.get('params_n_classes', K)),
        'ordinal_head': str(p.get('params_ordinal_head', 'CORAL')),
        'sequence_length': int(p.get('params_sequence_length', 12)),
        'hidden1': int(p.get('params_hidden1', 128)),
        'hidden2': int(p.get('params_hidden2', 64)),
        'num_layers': int(p.get('params_num_layers', 2)),
        'dropout': float(p.get('params_dropout', 0.3)),
        'inter_rnn_drop': float(p.get('params_inter_rnn_drop', 0.1)),
        'learning_rate': float(p.get('params_learning_rate', 1e-3)),
        'weight_decay': float(p.get('params_weight_decay', 1e-4)),
        'batch_size': int(p.get('params_batch_size', 32)),
        'max_epochs': int(p.get('params_max_epochs', 100)),
        'early_stopping_patience': int(p.get('params_early_stopping_patience', 15)),
        'early_stopping_min_delta': float(p.get('params_early_stopping_min_delta', 0.0)),
        'lr_patience': int(p.get('params_lr_patience', 7)),
        'lr_factor': float(p.get('params_lr_factor', 0.5)),
        'horizon_steps': int(p.get('params_horizon_steps', horizon)) if 'params_horizon_steps' in p else int(horizon),
        't_min': float(p.get('params_t_min', 0.6)),
        't_max': float(p.get('params_t_max', 2.5)),
        't_steps': int(p.get('params_t_steps', 15)),
        'tau_min': float(p.get('params_tau_min', 0.2)),
        'tau_max': float(p.get('params_tau_max', 0.8)),
        'tau_steps': int(p.get('params_tau_steps', 31)),
        'lambda_balance': float(p.get('params_lambda_balance', 0.05)),
        'coord_rounds': int(p.get('params_coord_rounds', 3)),
        'huber_delta': float(p.get('params_huber_delta', 1.0)),
    }

def select_and_apply_best_config(horizon=5, model='ALL', metric='mcc'):
    best_cfg = select_best_config(model=model, horizon=horizon, metric=metric)
    feature_set_name = best_cfg['feature_set']
    feature_cols = _select_feature_cols(feature_set_name)
    cfg = build_model_cfg(best_cfg, horizon)
    print(f"Using model={cfg.get('model_type')} feature_set={feature_set_name} with {len(feature_cols)} columns")
    return cfg, feature_cols, best_cfg


# In[37]:


master_df.reset_index(drop=True, inplace=True)
master_df


# In[38]:


h1_exit_df = pd.read_parquet(EARLY_EXIT_PATH)
h1_exit_df['date'] = pd.to_datetime(h1_exit_df['date']).dt.normalize()
master_df['date'] = pd.to_datetime(master_df['date']).dt.normalize()

print("Exit rows:", len(h1_exit_df))
print("Overlap tickers:", len(set(h1_exit_df['ticker']) & set(master_df['ticker'])))
print("Overlap dates:", len(set(h1_exit_df['date']) & set(master_df['date'])))



def count_early_exit_conditions(df, model):
    prob_col = f'p_up_1d_{model}'
    dir_col  = f'reg_dir_{model}'
    conf_col = f'reg_conf_{model}'

    missing = [c for c in [prob_col, dir_col, conf_col] if c not in df.columns]
    if missing:
        print(f"[{model}] Missing columns: {missing}")
        return None

    mask = (
        (df[prob_col] < 0.5) &
        (df[dir_col] == -1) &
        (df[conf_col] > 0.0)
    )
    count = int(mask.sum())
    total = int(df[[prob_col, dir_col, conf_col]].dropna().shape[0])
    frac = count / total if total > 0 else 0.0
    print(f"[{model}] count={count}, total={total}, fraction={frac:.4f}")
    return count, total, frac

for m in ['LSTM','BiLSTM','GRU','BiGRU']:
    count_early_exit_conditions(h1_exit_df, m)


# In[39]:


n_train_days = master_df['date'].nunique() - 100
print(f"Using initial training period of {n_train_days} days")


# # Execute Pipeline

# In[ ]:


initial_capital = 100_000.0
H_LIST = [1, 5]
MODEL_TYPE = "ALL"
METRIC = "mcc"
allow_overlap = True
seed = 42

output_dir = OUTPUT_ROOT / 'simulation' / 'multiclass' / 'meta'
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
        # model cfg (from multiclass benchmarking params)
        'model_feature_set': MODEL_CFG.get('feature_set') if 'MODEL_CFG' in globals() else None,
        'model_feature_group': MODEL_CFG.get('feature_group') if 'MODEL_CFG' in globals() else None,
        'model_mcc_mean': MODEL_CFG.get('mcc_mean') if 'MODEL_CFG' in globals() else None,
        'model_type': MODEL_CFG.get('model_type') if 'MODEL_CFG' in globals() else None,
        'model_n_classes': MODEL_CFG.get('n_classes') if 'MODEL_CFG' in globals() else None,
        'model_ordinal_head': MODEL_CFG.get('ordinal_head') if 'MODEL_CFG' in globals() else None,
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
        'model_early_stopping_patience': MODEL_CFG.get('early_stopping_patience') if 'MODEL_CFG' in globals() else None,
        'model_early_stopping_min_delta': MODEL_CFG.get('early_stopping_min_delta') if 'MODEL_CFG' in globals() else None,
        'model_lr_patience': MODEL_CFG.get('lr_patience') if 'MODEL_CFG' in globals() else None,
        'model_lr_factor': MODEL_CFG.get('lr_factor') if 'MODEL_CFG' in globals() else None,
        'model_horizon_steps': MODEL_CFG.get('horizon_steps') if 'MODEL_CFG' in globals() else None,
        'model_huber_delta': MODEL_CFG.get('huber_delta') if 'MODEL_CFG' in globals() else None,
        'model_t_min': MODEL_CFG.get('t_min') if 'MODEL_CFG' in globals() else None,
        'model_t_max': MODEL_CFG.get('t_max') if 'MODEL_CFG' in globals() else None,
        'model_t_steps': MODEL_CFG.get('t_steps') if 'MODEL_CFG' in globals() else None,
        'model_tau_min': MODEL_CFG.get('tau_min') if 'MODEL_CFG' in globals() else None,
        'model_tau_max': MODEL_CFG.get('tau_max') if 'MODEL_CFG' in globals() else None,
        'model_tau_steps': MODEL_CFG.get('tau_steps') if 'MODEL_CFG' in globals() else None,
        'model_lambda_balance': MODEL_CFG.get('lambda_balance') if 'MODEL_CFG' in globals() else None,
        'model_coord_rounds': MODEL_CFG.get('coord_rounds') if 'MODEL_CFG' in globals() else None,
    }

for H in H_LIST:
    MODEL_CFG, feature_columns, _best = select_and_apply_best_config(horizon=H, model=MODEL_TYPE, metric=METRIC)
    if 'MODEL_CFG' in globals():
        print('[CFG] tau grid:', f"t=[{MODEL_CFG.get('t_min')}, {MODEL_CFG.get('t_max')}]/{MODEL_CFG.get('t_steps')}, tau=[{MODEL_CFG.get('tau_min')}, {MODEL_CFG.get('tau_max')}]/{MODEL_CFG.get('tau_steps')}, lambda={MODEL_CFG.get('lambda_balance')}, coord_rounds={MODEL_CFG.get('coord_rounds')}")
    for early_exit_enabled in [True, False]:
        ee_tag = 'EE_ON' if early_exit_enabled else 'EE_OFF'
        for mode in ['KELLY_ONLY', 'HYBRID_KELLY_MPT', 'MPT_ONLY']:
            ALLOCATION_MODE = mode
            set_global_seeds(seed)
            print('=' * 80)
            print(f'Running allocation mode: {mode} | H={H} | {ee_tag}')
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


import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(simulation_results)


# In[ ]:


import numpy as np
import pandas as pd

all_trades = []
sector_lookup = None
if 'sector' in master_df.columns:
    sector_lookup = master_df.set_index(['ticker'])['sector'].to_dict()

for _, row in simulation_results.iterrows():
    exits = row.get('exits_today', [])
    if not isinstance(exits, list):
        continue
    for ex in exits:
        if not isinstance(ex, dict):
            continue
        ticker = ex.get('ticker')
        if ticker is None:
            continue
        allocated = ex.get('allocated_amount', ex.get('invest', np.nan))
        sector = ex.get('sector')
        if sector is None and sector_lookup is not None:
            sector = sector_lookup.get(ticker, 'UNKNOWN')
        prob_up = ex.get('prob_up', np.nan)
        expected_rise = ex.get('expected_rise', np.nan)
        exit_date = ex.get('exit_date')
        actual_return = ex.get('actual_return', np.nan)
        pnl = ex.get('pnl', np.nan)
        exit_reason = ex.get('exit_reason', 'unknown')
        all_trades.append({
            'date': ex.get('date', row.get('date')),
            'exit_date': exit_date,
            'ticker': ticker,
            'sector': sector if sector is not None else 'UNKNOWN',
            'allocated_amount': allocated,
            'prob_up': prob_up,
            'expected_rise': expected_rise,
            'actual_return': actual_return,
            'pnl': pnl,
            'exit_reason': exit_reason,
        })

trades_df = pd.DataFrame(all_trades)
print("="*80)
print("INVESTMENT SUMMARY")
print("="*80)
print(f"Total trades: {len(trades_df)}")
if trades_df.empty:
    print("No trades recorded.")
else:
    print(f"Total capital allocated: ${trades_df['allocated_amount'].sum():,.2f}")
    print(f"Average position size:  ${trades_df['allocated_amount'].mean():,.2f}")
    completed = trades_df[trades_df['pnl'].notna()].copy()
    print(f"Completed trades with returns: {len(completed)}")
    if not completed.empty:
        wins = (completed['pnl'] > 0).sum()
        losses = (completed['pnl'] < 0).sum()
        print(f"Winning trades: {wins} ({wins/len(completed)*100:.1f}%)")
        print(f"Losing trades:  {losses} ({losses/len(completed)*100:.1f}%)")
        total_pnl = completed['pnl'].sum()
        avg_pnl = completed['pnl'].mean()
        avg_ret = completed['actual_return'].mean()
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"Average PnL per trade: ${avg_pnl:,.2f}")
        print(f"Average return: {avg_ret*100:.2f}%")
    else:
        total_pnl = np.nan
        avg_pnl = np.nan
        avg_ret = np.nan

    if 'sector' in trades_df.columns:
        sector_stats = trades_df.groupby('sector').agg(
            trades=('ticker','count'),
            allocated=('allocated_amount','sum'),
            avg_alloc=('allocated_amount','mean'),
            pnl_sum=('pnl','sum'),
            pnl_mean=('pnl','mean'),
            ret_mean=('actual_return','mean'),
        ).sort_values('pnl_sum', ascending=False)
        print("Sector breakdown (sorted by total PnL):")
        print(sector_stats)

    if not trades_df.empty:
        print("TOP 10 BIGGEST EARNERS (trades)")
        print(trades_df.dropna(subset=['pnl']).sort_values('pnl', ascending=False).head(10))
        print("TOP 10 BIGGEST LOSERS (trades)")
        print(trades_df.dropna(subset=['pnl']).sort_values('pnl', ascending=True).head(10))

    if not trades_df.empty and 'sector' in trades_df.columns:
        top_sectors = trades_df.dropna(subset=['pnl']).groupby('sector')['pnl'].sum().sort_values(ascending=False)
        print("TOP 10 SECTORS BY TOTAL PnL")
        print(top_sectors.head(10).to_frame('total_pnl'))
        print("BOTTOM 10 SECTORS BY TOTAL PnL")
        print(top_sectors.tail(10).to_frame('total_pnl'))

        most_invested_tickers = trades_df['ticker'].value_counts().head(10)
        print("Top 10 Most Invested Tickers:")
        print(most_invested_tickers)

        all_tickers = master_df['ticker'].unique()
        invested_tickers = trades_df['ticker'].unique()
        never_invested_tickers = set(all_tickers) - set(invested_tickers)
        print(f"Tickers Never Invested In ({len(never_invested_tickers)}):")
        print(never_invested_tickers)


# In[ ]:


# folder_path = 'trained_models_multiclass_meta/'
# if os.path.exists(folder_path):
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')
# else:
#     print(f"Folder not found: {folder_path}")

# print(f"Contents of {folder_path} after deletion attempt:")
# if os.path.exists(folder_path):
#     print(os.listdir(folder_path))
# else:
#     print("Folder does not exist.")


# In[ ]:




