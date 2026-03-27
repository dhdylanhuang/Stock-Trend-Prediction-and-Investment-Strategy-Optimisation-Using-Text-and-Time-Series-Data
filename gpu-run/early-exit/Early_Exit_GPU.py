#!/usr/bin/env python
# coding: utf-8

# # Investment Simulation System

# ### Imports

# In[2]:


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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler


torch.__version__


# In[ ]:


import os
from pathlib import Path

PERSIST_ROOT = Path(os.environ.get('PERSIST_ROOT', '/mnt/primary'))
if not PERSIST_ROOT.exists():
    raise RuntimeError(f'Persistent storage not found at {PERSIST_ROOT}. Check mounts (df -h /mnt/primary).')

RUN_ROOT = Path(os.environ.get('RUN_ROOT', PERSIST_ROOT / 'early-exit'))
if not str(RUN_ROOT).startswith(str(PERSIST_ROOT)):
    print(f'WARNING: RUN_ROOT={RUN_ROOT} is not on persistent storage; forcing to {PERSIST_ROOT}/early-exit')
    RUN_ROOT = PERSIST_ROOT / 'early-exit'
RUN_ROOT.mkdir(parents=True, exist_ok=True)

MASTER_PATH = Path(os.environ.get('MASTER_PATH', RUN_ROOT / 'ta_nlp_sector.parquet'))
H1_EXIT_PATH = Path(os.environ.get('H1_EXIT_PATH', RUN_ROOT / 'h1_exit_df.parquet'))
MODEL_SAVE_PATH = Path(os.environ.get('MODEL_SAVE_PATH', RUN_ROOT / 'trained_models'))
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', RUN_ROOT / 'results'))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print('MASTER_PATH:', MASTER_PATH)
print('H1_EXIT_PATH:', H1_EXIT_PATH)
print('MODEL_SAVE_PATH:', MODEL_SAVE_PATH)
print('RESULTS_ROOT:', RESULTS_ROOT)

# DataLoader tuning for 4 CPU cores (adjust if needed)
NUM_WORKERS_TRAIN = 4
NUM_WORKERS_EVAL = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

def _dl_kwargs(num_workers: int):
    return dict(
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and num_workers > 0)
    )


# ### Configurations

# In[6]:


# MODEL_SAVE_PATH is set in the GPU path config cell
MIN_SEQUENCE_LENGTH = 12  # Minimum sequence length for any company
MAX_SEQUENCE_LENGTH = 12  # Maximum sequence length to cap computational cost
INITIAL_TRAINING_DAYS = 1100  # Number of days to use for initial training only
RETRAIN_INTERVAL = 60 # Retrain every 60 trading days (approx. quarterly)
MAX_DAY_GAP = 5  # Maximum allowed gap in trading days (to account for weekends/holidays)

# H=1 exit-monitor predictions (store probabilities only)
H_EXIT = 1
L_H1 = 12
MIN_SEQ = 50



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


# #### Define Horizon Target
# For a horizon H, compute both the direction and H-day return off the same base day t

# In[8]:


def add_horizon_targets(df: pd.DataFrame, H: int, price_col='close') -> pd.DataFrame:
    df = df.sort_values(['ticker','date']).copy()
    df[f'ret_{H}d']    = df.groupby('ticker')[price_col].shift(-H) / df[price_col] - 1.0
    df[f'target_{H}d'] = (df[f'ret_{H}d'] > 0).astype(int)
    return df


# #### Create Target Variable
# Build the binary classification target per row.

# In[9]:


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:

    print("Creating target variable...")
    df = df.sort_values(by=['ticker', 'date']).copy()
    df['next_day_close'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['next_day_close'] > df['close']).astype(int)
    df.dropna(subset=['next_day_close'], inplace=True)
    df['target'] = df['target'].astype(int)
    print("Target variable created.")
    return df


# ##### PyTorch Helpers

# In[10]:


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, fc=32, dropout=0.3, inter_dropout=0.1, use_layernorm=False):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_features, hidden_size=hidden1,
            num_layers=1, batch_first=True, bidirectional=False, dropout=0.1
        )
        self.inter_drop = nn.Dropout(p=inter_dropout)  # proxy for recurrent_dropout
        self.lstm2 = nn.LSTM(
            input_size=hidden1, hidden_size=hidden2,
            num_layers=1, batch_first=True, bidirectional=False, dropout=0.1
        )

        # Normalization after temporal pooling
        self.use_layernorm = bool(use_layernorm)
        if self.use_layernorm:
            self.norm = nn.LayerNorm(normalized_shape=hidden2)
        else:
            self.norm = nn.BatchNorm1d(num_features=hidden2)

        # Head
        self.fc1 = nn.Linear(hidden2, fc)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(fc, 1)  # logits

    def forward(self, x):
        # x: (B, T, F)
        out1, _ = self.lstm1(x)     # (B, T, hidden1)
        out1 = self.inter_drop(out1)
        out2, _ = self.lstm2(out1)  # (B, T, hidden2)

        last = out2[:, -1, :]       # (B, hidden2)
        if isinstance(self.norm, nn.BatchNorm1d):
            last = self.norm(last)  # BN expects (B, C)
        else:
            last = self.norm(last)  # LN expects (B, C)

        z = self.fc1(last)
        z = self.relu(z)
        z = self.drop(z)
        logits = self.fc_out(z).squeeze(-1)  # (B,)
        return logits

class EarlyStopper:
    def __init__(self, patience=15, mode='min'):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.best_state_dict = None
        self.mode = mode  # 'min' for val_loss
    def step(self, metric, model):
        if self.best_metric is None:
            improved = True
        else:
            improved = (metric < self.best_metric) if self.mode == 'min' else (metric > self.best_metric)
        if improved:
            self.best_metric = metric
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return improved

class RNNHead(nn.Module):
    def __init__(self, input_size, rnn_type='LSTM', bidirectional=False, problem_type='classification',
                 hidden1=128, hidden2=64, num_layers=1, inter_rnn_drop=0.1, dropout=0.3, use_layernorm=False):
        super().__init__()
        self.problem_type = problem_type
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.upper()
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
        out, _ = self.rnn1(x)
        if self.num_layers == 2:
            out = self.inter_rnn_drop(out)
            out, _ = self.rnn2(out)
        out = out[:, -1, :]
        out = self.bn(out)
        out = F.relu(self.fc(out))
        out = self.drop(out)
        out = self.out(out).squeeze(-1)
        return out


def build_model(input_shape, model_type='LSTM', problem_type='classification', hidden1=128, hidden2=64,
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


# In[11]:


@torch.no_grad()
def _evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_logits = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        all_logits.append(logits.detach().cpu())
    avg_loss = total_loss / len(loader.dataset)
    logits = torch.cat(all_logits, dim=0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    return avg_loss, probs


# In[12]:


folder_path = str(MODEL_SAVE_PATH)
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


# ### Load Data

# In[ ]:


# master_df = pd.read_parquet('stocknet-dataset/master_df.parquet')
master_df = pd.read_parquet(MASTER_PATH)


# In[14]:


columns_to_check = [
                    'sentiment',
                    
                    'emotion_anger', 'emotion_disgust', 'emotion_fear', 'emotion_joy',
                    'emotion_neutral', 'emotion_sadness', 'emotion_surprize',
                    'emotion_anger_pct', 'emotion_disgust_pct', 'emotion_fear_pct',
                    'emotion_joy_pct', 'emotion_neutral_pct', 'emotion_sadness_pct',
                    'emotion_surprize_pct', 
                    
                    'positive_emotion', 'negative_emotion','uncertainty_emotion', 
                    'positive_emotion_pct', 'negative_emotion_pct','uncertainty_emotion_pct', 
                    
                    'stance_label', 'stance_score', 
                    
                    'finbert_label', 'finbert_score', 'finbert_up', 'finbert_down',
                    'finbert_neutral', 
                    
                    'sector_open_mean', 'sector_high_mean', 'sector_low_mean', 'sector_close_mean',
                    'sector_volume_mean', 'sector_ret_1d', 'sector_ret_5d',
                    'sector_ret_20d', 'sector_range', 'sector_vol_20d', 
                    
                    'ema_12_sector','ema_26_sector', 'ema_50_sector', 'macd_12_26_9_sector',
                    'macdh_12_26_9_sector', 'macds_12_26_9_sector', 'rsi_14_sector',
                    'sector_bb_upper', 'sector_bb_middle', 'sector_bb_lower',
                    'market_close', 'sector_rel_strength', 'sector_dispersion_1d'
                ]

print(f"Initial master_df shape: {master_df.shape}")

master_df = master_df.dropna(subset=columns_to_check)

print(f"After dropping NaNs in selected columns, master_df shape: {master_df.shape}")

master_df.reset_index(drop=True, inplace=True)

print(master_df)


# In[15]:


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


# In[16]:


print(master_df.shape)
master_df = master_df.dropna(subset=feature_columns).sort_values(['ticker','date'])
print(master_df.shape)


# In[17]:


# Feature sets (exit) — map selected feature set to available columns
FEATURES_H1_EXIT = feature_columns
FEATURE_SET_ALIAS = { 'sentinment': 'sentiment' }

feature_sets = {
    'base': FEATURES_H1_EXIT,
    'sentiment': FEATURES_H1_EXIT,
    'emotion': FEATURES_H1_EXIT,
    'unified_emotion': FEATURES_H1_EXIT,
    'finbert': FEATURES_H1_EXIT,
    'all_nlp': FEATURES_H1_EXIT,
    'sector': FEATURES_H1_EXIT,
    'sector_sentiment': FEATURES_H1_EXIT,
    'sector_emotion': FEATURES_H1_EXIT,
    'sector_unified_emotion': FEATURES_H1_EXIT,
    'sector_finbert': FEATURES_H1_EXIT,
    'sector_all_nlp': FEATURES_H1_EXIT,
}

feature_groups = {
    'base': ['base'],
    'nlp': ['sentiment', 'emotion', 'unified_emotion', 'finbert', 'all_nlp'],
    'sector': ['sector', 'sector_sentiment', 'sector_emotion', 'sector_unified_emotion', 'sector_finbert', 'sector_all_nlp'],
}


feature_sets = {
    'base': FEATURES_H1_EXIT,
    'sentiment': FEATURES_H1_EXIT,
    'emotion': FEATURES_H1_EXIT,
    'unified_emotion': FEATURES_H1_EXIT,
    'finbert': FEATURES_H1_EXIT,
    'all_nlp': FEATURES_H1_EXIT,
    'sector': FEATURES_H1_EXIT,
    'sector_sentiment': FEATURES_H1_EXIT,
    'sector_emotion': FEATURES_H1_EXIT,
    'sector_unified_emotion': FEATURES_H1_EXIT,
    'sector_finbert': FEATURES_H1_EXIT,
    'sector_all_nlp': FEATURES_H1_EXIT,
}

feature_groups = {
    'base': ['base'],
    'nlp': ['sentiment', 'emotion', 'unified_emotion', 'finbert', 'all_nlp'],
    'sector': ['sector', 'sector_sentiment', 'sector_emotion', 'sector_unified_emotion', 'sector_finbert', 'sector_all_nlp'],
}


# Build targets
h1_df = add_horizon_targets(master_df.copy(), H=H_EXIT, price_col='close')
ret_col = f'ret_{H_EXIT}d'
tgt_col = f'target_{H_EXIT}d'

h1_df = h1_df.dropna(subset=FEATURES_H1_EXIT + [ret_col, tgt_col])
unique_dates = sorted(h1_df['date'].unique())
train_cutoff_date = unique_dates[INITIAL_TRAINING_DAYS - 1]
last_100_dates = set(unique_dates[-100:])
print(f"Train cutoff: {train_cutoff_date}")
print(f"Last 100 window: {min(last_100_dates)} -> {max(last_100_dates)}")
# Feature sets (exit) — map selected feature set to available columns
FEATURES_H1_EXIT = feature_columns
FEATURE_SET_ALIAS = { 'sentinment': 'sentiment' }


# In[18]:


def build_sequences_idx(df, feature_cols, target_col, seq_len):
    X_vals = df[feature_cols].values
    y_vals = df[target_col].values
    X_list, y_list, idx_list = [], [], []
    for i in range(seq_len - 1, len(df)):
        window = X_vals[i - seq_len + 1:i + 1]
        if np.isnan(window).any():
            continue
        if pd.isna(y_vals[i]):
            continue
        X_list.append(window)
        y_list.append(y_vals[i])
        idx_list.append(i)
    return np.array(X_list), np.array(y_list), idx_list




def eval_regression(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    preds = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda")):
                out = model(xb)
                loss = loss_fn(out, yb)
        total_loss += loss.item() * xb.size(0)
        preds.append(out.detach().cpu())
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(preds, dim=0).numpy().reshape(-1)
    return avg_loss, preds



def _resolve_feature_set_name(name: str):
    name = FEATURE_SET_ALIAS.get(name, name)
    if name not in feature_sets:
        raise KeyError(f"Unknown feature set '{name}'. Available: {sorted(feature_sets.keys())}")
    return name


def _select_feature_cols(feature_set_name: str):
    cols = feature_sets.get(feature_set_name, FEATURES_H1_EXIT)
    missing = [c for c in cols if c not in master_df.columns]
    if missing:
        print(f"[WARN] Missing cols for feature_set '{feature_set_name}'. Using FEATURES_H1_EXIT instead. Missing: {missing}")
        cols = FEATURES_H1_EXIT
    return cols


def select_best_configs(task='classification', metric='mcc', models=('LSTM','BiLSTM','GRU','BiGRU')):
    base_dir = RESULTS_ROOT / 'benchmarking' / task
    if not base_dir.exists():
        base_dir = Path('results/benchmarking') / task
    if not base_dir.exists():
        raise FileNotFoundError(f"No results directory found: {base_dir}")

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
            })

    if not rows:
        raise FileNotFoundError(f"No H=1 results/params found under {base_dir}")

    rows = sorted(rows, key=lambda r: r['score'], reverse=True)
    best = {}
    for r in rows:
        if r['model'] not in best:
            best[r['model']] = {
                'feature_group': r['feature_group'],
                'feature_set': r['feature_set'],
                'mcc_mean': r['score'],
                'params': r['params'],
            }
        if len(best) == len(models):
            break

    missing = [m for m in models if m not in best]
    if missing:
        raise FileNotFoundError(f"Missing best configs for models: {missing}")

    return best



# In[19]:


# Build targets
h1_df = add_horizon_targets(master_df.copy(), H=H_EXIT, price_col='close')
ret_col = f'ret_{H_EXIT}d'
tgt_col = f'target_{H_EXIT}d'

h1_df = h1_df.dropna(subset=FEATURES_H1_EXIT + [ret_col, tgt_col])
unique_dates = sorted(h1_df['date'].unique())
train_cutoff_date = unique_dates[INITIAL_TRAINING_DAYS - 1]
last_100_dates = set(unique_dates[-100:])
print(f"Train cutoff: {train_cutoff_date}")
print(f"Last 100 window: {min(last_100_dates)} -> {max(last_100_dates)}")
print(f"Samples in last 100-day window: {h1_df[h1_df['date'].isin(last_100_dates)].shape[0]:,}")

MODELS = ['LSTM','BiLSTM','GRU','BiGRU']
classification_best = select_best_configs('classification', models=MODELS)

regression_base_dir = RESULTS_ROOT / 'benchmarking' / 'regression'
if not regression_base_dir.exists():
    regression_base_dir = Path('results/benchmarking/regression')
if regression_base_dir.exists():
    regression_best = select_best_configs('regression', models=MODELS)
else:
    print('[WARN] No regression tuning results found; using classification configs for regression.')
    regression_best = classification_best

# Main walk-forward loop (per model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

all_preds = []  # list of per-model dfs

for problem_type, best_map in [('classification', classification_best), ('regression', regression_best)]:
    print(f"=== EARLY EXIT {problem_type.upper()} MODELS ===")
    for model in MODELS:
        cfg = best_map[model]
        params = cfg['params']
        feature_set_name = _resolve_feature_set_name(str(cfg['feature_set']))
        feature_cols = _select_feature_cols(feature_set_name)

        # hyperparameters (from params csv, with fallbacks)
        seq_len = int(params.get('params_sequence_length', L_H1))
        hidden1 = int(params.get('params_hidden1', 128))
        hidden2 = int(params.get('params_hidden2', 64))
        num_layers = int(params.get('params_num_layers', 2))
        inter_rnn_drop = float(params.get('params_inter_rnn_drop', 0.1))
        dropout = float(params.get('params_dropout', 0.3))
        learning_rate = float(params.get('params_learning_rate', 1e-3))
        weight_decay = float(params.get('params_weight_decay', 1e-4))
        huber_delta = float(params.get('params_huber_delta', 1.0))
        max_epochs = int(params.get('params_max_epochs', 100))
        early_patience = int(params.get('params_early_stopping_patience', 15))
        early_min_delta = float(params.get('params_early_stopping_min_delta', 0.0))
        lr_patience = int(params.get('params_lr_patience', 7))
        lr_factor = float(params.get('params_lr_factor', 0.5))
        batch_size = int(params.get('params_batch_size', 32))

        print(f"[{problem_type.upper()}] {model} | feature_set={feature_set_name} | seq={seq_len} | h1={hidden1} h2={hidden2} | layers={num_layers} | lr={learning_rate} wd={weight_decay} | bs={batch_size}")

        pred_rows = []
        for ticker, df_tkr in h1_df.groupby('ticker'):
            df_tkr = df_tkr.sort_values('date').reset_index(drop=True)
            df_tkr = df_tkr.dropna(subset=feature_cols + [ret_col, tgt_col])
            if len(df_tkr) < (seq_len + 10):
                continue

            target_col = tgt_col if problem_type == 'classification' else ret_col
            X_all, y_all, idx_list = build_sequences_idx(df_tkr, feature_cols, target_col, seq_len)
            if len(X_all) < MIN_SEQ:
                continue

            idx_arr = np.array(idx_list)
            idx_to_seq = {idx_list[i]: i for i in range(len(idx_list))}

            end_dates = df_tkr.iloc[idx_list]['date']
            pred_end_idxs = end_dates[(end_dates > train_cutoff_date) & (end_dates.isin(last_100_dates))].index.tolist()
            if not pred_end_idxs:
                continue
            pred_end_idxs = sorted(pred_end_idxs)

            pos = 0
            while pos < len(pred_end_idxs):
                block_start_idx = pred_end_idxs[pos]
                if block_start_idx < 2:
                    pos += RETRAIN_INTERVAL
                    continue
                train_end_idx = block_start_idx - 2  # avoid label leakage at t-1
                train_mask = idx_arr <= train_end_idx
                if train_mask.sum() < MIN_SEQ:
                    pos += RETRAIN_INTERVAL
                    continue

                X_train_full = X_all[train_mask]
                y_train_full = y_all[train_mask]
                split = max(int(len(X_train_full) * 0.8), 1)
                if split >= len(X_train_full):
                    pos += RETRAIN_INTERVAL
                    continue
                X_tr, y_tr = X_train_full[:split], y_train_full[:split]
                X_val, y_val = X_train_full[split:], y_train_full[split:]

                scaler = MinMaxScaler()
                n_feat = X_tr.shape[-1]
                X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
                X_val_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)

                train_ds = SequenceDataset(X_tr_s, y_tr)
                val_ds   = SequenceDataset(X_val_s, y_val)

                train_bs = min(batch_size, len(train_ds))
                if train_bs < 2:
                    pos += RETRAIN_INTERVAL
                    continue
                if len(train_ds) % train_bs == 1 and train_bs > 2:
                    train_bs -= 1
                val_bs = min(batch_size, len(val_ds))

                train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False, drop_last=False, **_dl_kwargs(NUM_WORKERS_TRAIN))
                val_loader   = DataLoader(val_ds,   batch_size=val_bs, shuffle=False, drop_last=False, **_dl_kwargs(NUM_WORKERS_EVAL))

                model_t = build_model((seq_len, n_feat), model_type=model, problem_type=problem_type, hidden1=hidden1, hidden2=hidden2, num_layers=num_layers, inter_rnn_drop=inter_rnn_drop, dropout=dropout).to(device)

                if problem_type == 'classification':
                    loss_fn = nn.BCEWithLogitsLoss()
                else:
                    loss_fn = nn.HuberLoss(delta=huber_delta)

                optimizer = torch.optim.Adam(model_t.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=1e-7)
                early = EarlyStopper(patience=early_patience, mode='min')

                for epoch in range(max_epochs):
                    model_t.train()
                    total_loss = 0.0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = model_t(xb)
                        loss = loss_fn(logits, yb)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model_t.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item() * xb.size(0)

                    if problem_type == 'classification':
                        val_loss, val_probs = _evaluate(model_t, val_loader, device, loss_fn)
                    else:
                        val_loss, val_probs = eval_regression(model_t, val_loader, device, loss_fn)

                    scheduler.step(val_loss)
                    _ = early.step(val_loss, model_t)
                    if early.counter >= early.patience:
                        break

                if early.best_state_dict is not None:
                    model_t.load_state_dict(early.best_state_dict)

                block_indices = pred_end_idxs[pos:pos + RETRAIN_INTERVAL]
                X_pred = []
                pred_dates = []
                for idx in block_indices:
                    seq_idx = idx_to_seq.get(idx)
                    if seq_idx is None:
                        continue
                    X_pred.append(X_all[seq_idx])
                    pred_dates.append(df_tkr.iloc[idx]['date'])

                if X_pred:
                    X_pred = np.stack(X_pred)
                    X_pred_s = scaler.transform(X_pred.reshape(-1, n_feat)).reshape(X_pred.shape)
                    with torch.no_grad():
                        xb = torch.from_numpy(X_pred_s.astype(np.float32)).to(device)
                        logits = model_t(xb)
                        raw = logits.detach().cpu().numpy().reshape(-1)

                    if problem_type == 'classification':
                        raw_probs = 1.0 / (1.0 + np.exp(-raw))
                        pred_min, pred_max = raw_probs.min(), raw_probs.max()
                        buffer = 0.05 * (pred_max - pred_min)
                        y_min_dynamic = max(0.0, pred_min - buffer)
                        y_max_dynamic = min(1.0, pred_max + buffer)
                        calibrator = IsotonicRegression(y_min=y_min_dynamic, y_max=y_max_dynamic, out_of_bounds='clip')
                        # Fit on validation probs; apply to prediction probs
                        calibrator.fit(val_probs, y_val.astype(float))
                        cal_probs = calibrator.predict(raw_probs)

                        for d, p in zip(pred_dates, cal_probs):
                            pred_rows.append({
                                'date': pd.to_datetime(d),
                                'ticker': ticker,
                                f'p_up_1d_{model}': float(np.clip(p, 0.0, 1.0)),
                            })
                    else:
                        for d, p in zip(pred_dates, raw):
                            pred_rows.append({
                                'date': pd.to_datetime(d),
                                'ticker': ticker,
                                f'reg_dir_{model}': 1 if p > 0 else -1,
                                f'reg_conf_{model}': float(abs(p)),
                            })

                pos += RETRAIN_INTERVAL

        df_preds = pd.DataFrame(pred_rows)
        if df_preds.empty:
            print(f"[WARN] No predictions for {problem_type} {model}")
            continue
        df_preds = df_preds.drop_duplicates(subset=['date','ticker'])
        df_preds = df_preds.sort_values(['date','ticker']).reset_index(drop=True)
        all_preds.append(df_preds)

# Merge all predictions
if all_preds:
    merged = all_preds[0]
    for dfp in all_preds[1:]:
        merged = merged.merge(dfp, on=['date','ticker'], how='outer')
else:
    merged = pd.DataFrame(columns=['date','ticker'])

# keep only last 100 dates for simulator
before_rows = len(merged)
merged = merged[merged['date'].isin(last_100_dates)].copy()
after_rows = len(merged)
print(f"Filtered to last 100 dates: {before_rows} -> {after_rows} rows")
merged = merged.sort_values(['date', 'ticker']).reset_index(drop=True)

print('early_exit_df shape:', merged.shape)
if not merged.empty:
    # pick best classification model for convenience
    best_cls_model = max(classification_best, key=lambda m: classification_best[m]['mcc_mean'])
    col = f'p_up_1d_{best_cls_model}'
    if col in merged.columns:
        merged['p_up_1d'] = merged[col]

out_path = H1_EXIT_PATH
merged.to_parquet(out_path, index=False)
print(f'Saved: {out_path}')

