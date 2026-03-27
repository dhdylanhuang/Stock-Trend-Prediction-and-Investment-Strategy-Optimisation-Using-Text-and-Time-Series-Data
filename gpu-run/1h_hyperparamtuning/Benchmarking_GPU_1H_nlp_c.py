#!/usr/bin/env python
# coding: utf-8

# # Benchmarking Results from Classification and Regression

# #### Set Up

# In[1]:


import pandas as pd
import numpy as np
import site
import os


# In[ ]:


# Preprocessing cache to avoid recomputing features/sequences/splits per trial
from collections import OrderedDict

USE_PREPROCESS_CACHE = True
CACHE_MAX_ITEMS = 256  # adjust if memory is tight
PREPROCESS_CACHE = OrderedDict()

def _cache_get(key):
    if not USE_PREPROCESS_CACHE:
        return None
    val = PREPROCESS_CACHE.get(key)
    if val is not None:
        PREPROCESS_CACHE.move_to_end(key)
        try:
            print(f"[CACHE HIT] key={key}")
        except Exception:
            print("[CACHE HIT] (key repr failed)")
    return val

def _cache_set(key, val):
    if not USE_PREPROCESS_CACHE:
        return
    PREPROCESS_CACHE[key] = val
    PREPROCESS_CACHE.move_to_end(key)
    if len(PREPROCESS_CACHE) > CACHE_MAX_ITEMS:
        PREPROCESS_CACHE.popitem(last=False)


# In[ ]:


# DataLoader tuning for 4 CPU cores
NUM_WORKERS_TRAIN = 1
NUM_WORKERS_EVAL = 1
PIN_MEMORY = True
PERSISTENT_WORKERS = True


# In[ ]:


from pathlib import Path

PERSIST_ROOT = Path(os.environ.get('PERSIST_ROOT', '/mnt/primary'))
if not PERSIST_ROOT.exists():
    raise RuntimeError(f'Persistent storage not found at {PERSIST_ROOT}. Check mounts (df -h /mnt/primary).')

RUN_ROOT = Path(os.environ.get('RUN_ROOT', PERSIST_ROOT / '1h_hyperparamtuning'))
if not str(RUN_ROOT).startswith(str(PERSIST_ROOT)):
    print(f'WARNING: RUN_ROOT={RUN_ROOT} is not on persistent storage; forcing to {PERSIST_ROOT}/1h_hyperparamtuning')
    RUN_ROOT = PERSIST_ROOT / '1h_hyperparamtuning'
RUN_ROOT.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(os.environ.get('DATA_PATH', RUN_ROOT / 'master_df_60rf.parquet'))
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', RUN_ROOT / 'meta-results' / 'benchmarking'))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

print('DATA_PATH:', DATA_PATH)
print('RESULTS_ROOT:', RESULTS_ROOT)


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from contextlib import nullcontext

import random

from unicodedata import bidirectional


# ### Utility Classes and Functions

# In[4]:


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
                 hidden1=128, hidden2=64, num_layers=1,
                 inter_rnn_drop=0.1, dropout=0.3, use_layernorm=False):
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


def build_model(input_shape, model_type='LSTM', problem_type='regression', hidden1=128, hidden2=64, 
                num_layers=2, inter_rnn_drop=0.1, dropout=0.3, use_layernorm=False):
    seq_len, n_features = input_shape
    model_type = model_type.upper()
    kwargs = dict(
        problem_type=problem_type,
        hidden1=hidden1,
        hidden2=hidden2,
        num_layers=num_layers,
        inter_rnn_drop=inter_rnn_drop,
        dropout=dropout,
        use_layernorm=use_layernorm,
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


# In[5]:


def pct_return(series, h):
    return series.shift(-h) / series - 1.0

def best_threshold_from_val(y_true, y_scores, metric='f1', grid=None):
    """
    Sweep probability thresholds on validation scores to maximize a metric.
    metric can be 'f1', 'mcc', 'accuracy', or a callable(y_true,y_pred)->float.
    Returns (best_threshold, best_metric_value).
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    metric_fn = None
    if callable(metric):
        metric_fn = metric
    else:
        name = str(metric).lower()
        if name == 'f1':
            metric_fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)
        elif name == 'mcc':
            metric_fn = lambda yt, yp: matthews_corrcoef(yt, yp)
        elif name in ('acc', 'accuracy'):
            metric_fn = lambda yt, yp: (yt == yp).mean()
        else:
            raise ValueError(f"Unsupported metric '{metric}'")
    best_thr = 0.5
    best_val = -np.inf
    for thr in grid:
        preds = (y_scores >= thr).astype(int)
        val = metric_fn(y_true, preds)
        if val > best_val + 1e-12 or (abs(val - best_val) <= 1e-12 and thr < best_thr):
            best_val = float(val)
            best_thr = float(thr)
    return best_thr, best_val


# ## Stock Prediction Pipeline

# In[ ]:


class StockPredictionPipeline:
    def __init__(self, df, feature_columns, model_type='LSTM', sequence_length=20, problem_type='regression', horizon_steps=1,
                 hidden1=64, hidden2=128, num_layers=1, inter_rnn_drop=0.4, dropout=0.7,
                 batch_size=16, learning_rate=4e-3, weight_decay=0, lr_patience=7, lr_factor=0.5,
                 early_stopping_patience=15, max_epochs=20, use_layernorm=False, huber_delta=1.66, 
                 early_stopping_min_delta=0.0, trial=None):
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.problem_type = problem_type
        self.horizon_steps = horizon_steps
        self.results = []
        self.loss_curves = []
        self.skip_reasons = {}
        self.skipped_companies = []
        self.trial = trial
        self._trial_step = 0
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_layers = num_layers
        self.inter_rnn_drop = inter_rnn_drop
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.use_layernorm = use_layernorm
        self.huber_delta = huber_delta
        self.early_stopping_min_delta = early_stopping_min_delta

        # Validate
        self._validate_inputs()

        # Device & precision
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.mixed_precision = (self.device.type == 'cuda')

        print(f"Pipeline initialized for a '{self.problem_type}' problem "
              f"with horizon {self.horizon_steps} steps. Device: {self.device}")

    def _validate_inputs(self):
        missing_cols = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        if not any(col in self.df.columns for col in ['adj_close', 'adj close', 'close', 'close_price']):
            raise ValueError("No 'adj_close'/'adj close'/'close' or 'close_price' column found in data")

        valid_models = ['LSTM', 'BiLSTM', 'GRU', 'BiGRU']
        if self.model_type not in valid_models:
            raise ValueError(f"Model type must be one of: {valid_models}")

        if self.problem_type not in ['regression', 'classification']:
            raise ValueError("Problem type must be 'regression' or 'classification'")

    def _record_skip(self, company_name, reason, sector=None):
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
        self.skipped_companies.append({'company': company_name, 'sector': sector, 'reason': reason})

    def create_target_variable(self, company_data):
        company_data = company_data.copy()
        price_col = 'close' if 'close' in company_data.columns else 'close_price'
        if 'date' in company_data.columns:
            company_data = company_data.sort_values('date')

        h = self.horizon_steps

        company_data['target_regression'] = (
            np.log(company_data[price_col].shift(-h)) - np.log(company_data[price_col])
        )
        company_data['target_direction'] = (company_data['target_regression'] > 0).astype(int)
        company_data['ret_h'] = pct_return(company_data[price_col], h)
        company_data = company_data.dropna()
        return company_data

    def create_sequences(self, features, *targets):
        X = []
        y_sequences = [[] for _ in targets]
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            for j, target in enumerate(targets):
                y_sequences[j].append(target[i])
        return (np.array(X),) + tuple(np.array(y) for y in y_sequences)

    def _train_one_epoch(self, model, loader, optimizer, loss_fn, scaler):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            ctx = torch.amp.autocast('cuda') if self.mixed_precision else nullcontext()
            with ctx:
                logits = model(xb)
                loss = loss_fn(logits, yb)

            if self.mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_one_epoch(self, model, loader, loss_fn):
        model.eval()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device).view(-1, 1)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _predict(self, model, loader):
        model.eval()
        outs = []
        for xb, _ in loader:
            xb = xb.to(self.device)
            logits = model(xb).squeeze(1).detach().cpu().numpy()
            outs.append(logits)
        return np.concatenate(outs, axis=0)

    def build_model(self, input_shape):
        model = build_model(
            input_shape,
            model_type=self.model_type,
            problem_type=self.problem_type,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            num_layers=self.num_layers,
            inter_rnn_drop=self.inter_rnn_drop,
            dropout=self.dropout,
            use_layernorm=self.use_layernorm
        )
        return model.to(self.device)

    def _empty_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps' and hasattr(torch, 'mps'):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def process_company(self, company_name, company_data, sector):
        print(f"\nProcessing {company_name} ({sector})...")
        try:
            # Fixed date windows (keep consistent with diagnostics)
            train_start = pd.Timestamp('2014-01-01')
            train_end = pd.Timestamp('2015-08-01')
            val_end = pd.Timestamp('2015-10-01')
            test_end = pd.Timestamp('2016-01-01')

            cache_key = (
                company_name,
                tuple(self.feature_columns),
                self.sequence_length,
                self.horizon_steps,
                train_start,
                train_end,
                val_end,
                test_end,
            )
            cached = _cache_get(cache_key)

            if cached is None:
                company_data = company_data.copy()
                if 'date' not in company_data.columns:
                    print(f"Missing date column for {company_name}. Skipping...")
                    self._record_skip(company_name, "missing_date", sector)
                    return None
                company_data['date'] = pd.to_datetime(company_data['date'], errors='coerce')
                company_data = company_data.dropna(subset=['date'])
                if company_data.empty:
                    print(f"No valid dates for {company_name}. Skipping...")
                    self._record_skip(company_name, "invalid_date", sector)
                    return None
                company_data = self.create_target_variable(company_data)

                # Min samples requirement (same heuristic)
                min_samples = self.sequence_length + self.horizon_steps + 60
                if len(company_data) < min_samples:
                    print(f"Insufficient data for {company_name} ({len(company_data)} < {min_samples}). Skipping...")
                    self._record_skip(company_name, "insufficient_data", sector)
                    return None

                if company_data[self.feature_columns].isnull().any().any():
                    print(f"Missing values in features for {company_name}. Skipping...")
                    self._record_skip(company_name, "missing_features", sector)
                    return None

                features = company_data[self.feature_columns].values
                target_reg = company_data['target_regression'].values
                target_dir = company_data['target_direction'].values

                # Create sequences
                X_raw, y_reg, y_dir = self.create_sequences(features, target_reg, target_dir)

                # Date-based split (fixed ranges)
                seq_end_dates = pd.to_datetime(company_data['date'].values[self.sequence_length:], errors='coerce')
                valid_dates_mask = ~pd.isna(seq_end_dates)
                if not valid_dates_mask.any():
                    print(f"No valid sequence dates for {company_name}. Skipping...")
                    self._record_skip(company_name, "no_seq_dates", sector)
                    return None

                # Apply valid date mask to sequences and targets
                X_raw = X_raw[valid_dates_mask]
                y_reg = y_reg[valid_dates_mask]
                y_dir = y_dir[valid_dates_mask]
                seq_end_dates = seq_end_dates[valid_dates_mask]

                train_mask = (seq_end_dates >= train_start) & (seq_end_dates < train_end)
                val_mask   = (seq_end_dates >= train_end) & (seq_end_dates < val_end)
                test_mask  = (seq_end_dates >= val_end) & (seq_end_dates < test_end)

                train_idx = np.where(train_mask)[0]
                val_idx   = np.where(val_mask)[0]
                test_idx  = np.where(test_mask)[0]

                if len(train_idx) == 0:
                    print(f"No training data in range for {company_name}. Skipping...")
                    self._record_skip(company_name, "no_train_window", sector)
                    return None
                if len(val_idx) == 0:
                    print(f"No validation data in range for {company_name}. Skipping...")
                    self._record_skip(company_name, "no_val_window", sector)
                    return None
                if len(test_idx) == 0:
                    print(f"No test data in range for {company_name}. Skipping...")
                    self._record_skip(company_name, "no_test_window", sector)
                    return None

                X_train_raw, X_val_raw, X_test_raw = X_raw[train_idx], X_raw[val_idx], X_raw[test_idx]

                F = X_raw.shape[-1]
                feat_scaler = StandardScaler()
                X_train = feat_scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(X_train_raw.shape)
                X_val   = feat_scaler.transform(X_val_raw.reshape(-1, F)).reshape(X_val_raw.shape)
                X_test  = feat_scaler.transform(X_test_raw.reshape(-1, F)).reshape(X_test_raw.shape)

                y_reg_train, y_reg_val, y_reg_test = y_reg[train_idx], y_reg[val_idx], y_reg[test_idx]
                y_dir_train, y_dir_val, y_dir_test = y_dir[train_idx], y_dir[val_idx], y_dir[test_idx]

                # Always fit target scaler once so regression can reuse it across trials
                target_scaler = StandardScaler()
                y_reg_train_scaled = target_scaler.fit_transform(y_reg_train.reshape(-1, 1)).flatten()
                y_reg_val_scaled   = target_scaler.transform(y_reg_val.reshape(-1, 1)).flatten()

                n_samples = int(X_raw.shape[0])

                cached = {
                    'X_train': X_train,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_reg_train': y_reg_train,
                    'y_reg_val': y_reg_val,
                    'y_reg_test': y_reg_test,
                    'y_dir_train': y_dir_train,
                    'y_dir_val': y_dir_val,
                    'y_dir_test': y_dir_test,
                    'y_reg_train_scaled': y_reg_train_scaled,
                    'y_reg_val_scaled': y_reg_val_scaled,
                    'target_scaler': target_scaler,
                    'n_samples': n_samples,
                }
                _cache_set(cache_key, cached)

            X_train = cached['X_train']
            X_val = cached['X_val']
            X_test = cached['X_test']
            y_reg_train = cached['y_reg_train']
            y_reg_val = cached['y_reg_val']
            y_reg_test = cached['y_reg_test']
            y_dir_train = cached['y_dir_train']
            y_dir_val = cached['y_dir_val']
            y_dir_test = cached['y_dir_test']
            y_reg_train_scaled = cached['y_reg_train_scaled']
            y_reg_val_scaled = cached['y_reg_val_scaled']
            target_scaler = cached['target_scaler']
            n_samples = cached['n_samples']

            if self.problem_type == 'regression':
                y_train, y_val, y_test = y_reg_train, y_reg_val, y_reg_test
                train_target, val_target = y_reg_train_scaled, y_reg_val_scaled
            else:
                y_train, y_val, y_test = y_dir_train, y_dir_val, y_dir_test
                train_target, val_target = y_train, y_val
            # class balance note
            if self.problem_type == 'classification':
                class_ratio = np.mean(y_train)
                if class_ratio < 0.1 or class_ratio > 0.9:
                    print(f"Severe class imbalance for {company_name} ({class_ratio:.3f}). Consider using class weights.")

            # datasets & loaders
            train_ds = SequenceDataset(X_train, train_target)
            val_ds   = SequenceDataset(X_val,   val_target)
            test_ds  = SequenceDataset(X_test,  y_test)

            train_bs = min(self.batch_size, len(train_ds))
            if train_bs < 2:
                print(f'Insufficient training samples for {company_name} (train size={len(train_ds)}). Skipping...')
                self._record_skip(company_name, "train_batch_too_small", sector)
                return None
            if len(train_ds) % train_bs == 1 and train_bs > 2:
                train_bs -= 1  # avoid batch size 1 for BatchNorm
            val_bs = min(self.batch_size, len(val_ds))
            test_bs = min(self.batch_size, len(test_ds))

            train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=False,  drop_last=False, num_workers=NUM_WORKERS_TRAIN, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)
            val_loader   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False, drop_last=False, num_workers=NUM_WORKERS_EVAL, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)
            test_loader  = DataLoader(test_ds,  batch_size=test_bs,  shuffle=False, drop_last=False, num_workers=NUM_WORKERS_EVAL, persistent_workers=PERSISTENT_WORKERS, pin_memory=PIN_MEMORY)

            # build model
            model = self.build_model((self.sequence_length, len(self.feature_columns)))

            # loss functions
            if self.problem_type == 'regression':
                loss_fn = nn.HuberLoss(delta=self.huber_delta)
            else:
                # use BCEWithLogitsLoss for numerical stability (logits input)
                loss_fn = nn.BCEWithLogitsLoss()

            # optimizer & scheduler
            optimizer = Adam(model.parameters(), lr=self.learning_rate, eps=1e-7, weight_decay=self.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience, min_lr=1e-7)
            early_stopper = EarlyStopper(patience=self.early_stopping_patience, min_delta=self.early_stopping_min_delta, restore_best=True)
            scaler = torch.amp.GradScaler('cuda', enabled=self.mixed_precision)

            # training loop
            max_epochs = self.max_epochs
            best_val = float('inf')
            epochs_trained = 0
            company_loss_rows = []

            for epoch in range(1, max_epochs + 1):
                train_loss = self._train_one_epoch(model, train_loader, optimizer, loss_fn, scaler)
                val_loss = self._eval_one_epoch(model, val_loader, loss_fn)
                scheduler.step(val_loss)
                stop = early_stopper.step(val_loss, model)
                epochs_trained = epoch


                row = {
                    'company': company_name,
                    'sector': sector,
                    'model_type': self.model_type,
                    'problem_type': self.problem_type,
                    'sequence_length': self.sequence_length,
                    'horizon_steps': self.horizon_steps,
                    'epoch': epoch,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                }

                company_loss_rows.append(row)
                self.loss_curves.append(row)

                if epoch % 10 == 0 or stop:
                    print(f"  Epoch {epoch:03d} - train {train_loss:.5f} | val {val_loss:.5f}")

                if stop:
                    break

            # restore best model weights (like Keras restore_best_weights=True)
            early_stopper.restore(model)

            # summarize train/val loss for overfitting checks
            best_train_loss = np.nan
            best_val_loss = np.nan
            final_train_loss = np.nan
            final_val_loss = np.nan
            if company_loss_rows:
                best_val_loss = min(r['val_loss'] for r in company_loss_rows)
                best_train_loss = min(r['train_loss'] for r in company_loss_rows)
                final_train_loss = company_loss_rows[-1]['train_loss']
                final_val_loss = company_loss_rows[-1]['val_loss']

            # predictions
            y_pred_raw = self._predict(model, test_loader)  # raw/regression or logits

            if self.problem_type == 'regression':
                y_pred_unscaled = target_scaler.inverse_transform(y_pred_raw.reshape(-1,1)).flatten() if target_scaler is not None else y_pred_raw
                mse = mean_squared_error(y_test, y_pred_unscaled)
                mae = mean_absolute_error(y_test, y_pred_unscaled)
                r2  = r2_score(y_test, y_pred_unscaled)

                # directional metrics (derived)
                y_test_dir = (y_reg_test > 0).astype(int)
                y_pred_dir = (y_pred_unscaled > 0).astype(int)
            else:
                # logits -> probs via sigmoid -> learn best threshold on VAL
                val_logits = self._predict(model, val_loader)
                val_probs = 1.0 / (1.0 + np.exp(-val_logits))
                best_thr, best_thr_score = best_threshold_from_val(y_val, val_probs, metric='mcc')
                val_pred_dir = (val_probs >= best_thr).astype(int)
                val_precision = precision_score(y_val, val_pred_dir, zero_division=0)
                val_recall = recall_score(y_val, val_pred_dir, zero_division=0)
                val_f1 = f1_score(y_val, val_pred_dir, zero_division=0)
                val_mcc = matthews_corrcoef(y_val, val_pred_dir)
                val_directional_accuracy = (y_val == val_pred_dir).mean()
                probs = 1.0 / (1.0 + np.exp(-y_pred_raw))
                y_pred_dir = (probs >= best_thr).astype(int)
                y_test_dir = y_test
                mse = mae = r2 = np.nan

            precision = precision_score(y_test_dir, y_pred_dir, zero_division=0)
            recall    = recall_score(y_test_dir, y_pred_dir, zero_division=0)
            f1        = f1_score(y_test_dir, y_pred_dir, zero_division=0)
            mcc       = matthews_corrcoef(y_test_dir, y_pred_dir)
            directional_accuracy = np.mean(y_test_dir == y_pred_dir)

            result = {
                'company': company_name,
                'sector': sector,
                'model_type': self.model_type,
                'problem_type': self.problem_type,
                'horizon_steps': self.horizon_steps,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'mcc': mcc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'directional_accuracy': directional_accuracy,
                'val_directional_accuracy': val_directional_accuracy if self.problem_type == 'classification' else np.nan,
                'val_mcc': val_mcc if self.problem_type == 'classification' else np.nan,
                'val_f1': val_f1 if self.problem_type == 'classification' else np.nan,
                'val_precision': val_precision if self.problem_type == 'classification' else np.nan,
                'val_recall': val_recall if self.problem_type == 'classification' else np.nan,
                'n_samples': int(n_samples),
                'train_samples': int(X_train.shape[0]),
                'val_samples': int(X_val.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'epochs_trained': epochs_trained
            }
            if self.problem_type == 'classification':
                result['best_threshold'] = best_thr
                result['best_threshold_metric'] = best_thr_score

            if self.problem_type == 'regression':
                print(f"  Regression -> MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
            elif self.problem_type == 'classification':
                print(f"  Classification -> best τ={best_thr:.3f} (val F1={best_thr_score:.4f})")
            print(f"  Directional -> Accuracy: {directional_accuracy:.4f}, MCC: {mcc:.4f}, F1: {f1:.4f}")

            # explicit cleanup (PyTorch handles this, but keeps parity with Enrique2025)
            del model
            self._empty_cache()

            return result

        except Exception as e:
            print(f"Error processing {company_name}: {str(e)}")
            self._record_skip(company_name, f"error:{type(e).__name__}", sector)
            self._empty_cache()
            return None

    def run_pipeline(self):
        company_col = None
        for col_name in ['ticker', 'company', 'symbol']:
            if col_name in self.df.columns:
                company_col = col_name
                break
        if company_col is None:
            company_col = self.df.columns[0]
            print(f"Warning: Using '{company_col}' as company identifier column")

        companies = self.df[company_col].unique()
        print(f"Processing {len(companies)} companies with {self.model_type} model...")
        print(f"Problem type: {self.problem_type}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Features: {self.feature_columns}")

        successful_companies = 0
        for i, company in enumerate(companies, 1):
            print(f"\n[{i}/{len(companies)}] Processing {company}...")
            company_data = self.df[self.df[company_col] == company].copy()
            sector = company_data['sector'].iloc[0] if 'sector' in company_data.columns else 'Unknown'
            result = self.process_company(company, company_data, sector)
            if result:
                self.results.append(result)
                successful_companies += 1

        print(f"\n{'='*80}")
        print(f"Pipeline completed: {successful_companies}/{len(companies)} companies processed successfully")
        print(f"{'='*80}")
        if self.skip_reasons:
            print("Skip summary:")
            for reason, count in sorted(self.skip_reasons.items(), key=lambda x: (-x[1], x[0])):
                print(f"  {reason}: {count}")
        if self.skipped_companies:
            self.skipped_df = pd.DataFrame(self.skipped_companies)

        if self.results:
            self.results_df = pd.DataFrame(self.results)
            return self.results_df
        else:
            print("No companies were processed successfully!")
            return pd.DataFrame()


    def analyze_results(self):
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("No results to analyze!")
            return None

        df = self.results_df
        analysis = {}

        print("" + "="*80)
        print("STOCK PREDICTION PIPELINE RESULTS")
        print("="*80)
        print(f"Model: {self.model_type} | Problem: {self.problem_type}")
        print(f"Companies analyzed: {len(df)}")
        print(f"Average samples per company: {df['n_samples'].mean():.0f}")

        print("" + "="*50)
        print("OVERALL PERFORMANCE")
        print("="*50)
        if self.problem_type == 'regression':
            print(f"Mean Squared Error:     {df['mse'].mean():.6f} (±{df['mse'].std():.6f})")
            print(f"Mean Absolute Error:    {df['mae'].mean():.6f} (±{df['mae'].std():.6f})")
            print(f"R² Score:              {df['r2'].mean():.4f} (±{df['r2'].std():.4f})")

        print(f"Directional Accuracy:   {df['directional_accuracy'].mean():.4f} (±{df['directional_accuracy'].std():.4f})")
        print(f"Matthews Correlation:   {df['mcc'].mean():.4f} (±{df['mcc'].std():.4f})")
        print(f"F1 Score:              {df['f1'].mean():.4f} (±{df['f1'].std():.4f})")
        print(f"Precision:             {df['precision'].mean():.4f} (±{df['precision'].std():.4f})")
        print(f"Recall:                {df['recall'].mean():.4f} (±{df['recall'].std():.4f})")

        if 'sector' in df.columns and df['sector'].nunique() > 1:
            print("" + "="*50)
            print("PERFORMANCE BY SECTOR")
            print("="*50)
            sector_stats = df.groupby('sector').agg({
                'directional_accuracy': ['mean', 'std', 'count'],
                'mcc': ['mean', 'std'],
                'r2': 'mean' if self.problem_type == 'regression' else lambda x: np.nan,
                'mae': 'mean' if self.problem_type == 'regression' else lambda x: np.nan
            }).round(4)
            sector_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in sector_stats.columns]
            sector_stats = sector_stats.sort_values('directional_accuracy_mean', ascending=False)
            for sector, row in sector_stats.iterrows():
                print(f"{sector:<20} | Acc: {row['directional_accuracy_mean']:.3f}±{row['directional_accuracy_std']:.3f} | "
                      f"MCC: {row['mcc_mean']:.3f} | Companies: {int(row['directional_accuracy_count'])}")

        print("" + "="*50)
        print("TOP 10 PERFORMERS (by Directional Accuracy)")
        print("="*50)
        top_performers = df.nlargest(10, 'directional_accuracy')
        for _, row in top_performers.iterrows():
            print(f"{row['company']:<20} | {row['sector']:<15} | "
                  f"Acc: {row['directional_accuracy']:.3f} | MCC: {row['mcc']:.3f}")

        return analysis

    def save_results(self, results, output_dir=None):
        if results is not None and not results.empty:
            model_name = self.model_type

            if output_dir is None:
                output_dir = f"results/benchmarking/base/{self.horizon_steps}H/"

            if self.problem_type == 'regression':
                out_dir = os.path.join(output_dir, 'regression')
            elif self.problem_type == 'classification':
                out_dir = os.path.join(output_dir, 'classification')

            os.makedirs(out_dir, exist_ok=True)

            output_path = os.path.join(out_dir, f"{model_name}.csv")

            results.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            print("No results to save.")

    def get_loss_curves_df(self):
        if not self.loss_curves:
            print("No loss curves logged yet.")
            return pd.DataFrame()
        return pd.DataFrame(self.loss_curves)

    def save_loss_curves(self, out_path=None):
        if out_path is None:
            out_path = f"results/benchmarking/base/{self.horizon_steps}H/"
        df = self.get_loss_curves_df()
        if df.empty:
            print("No loss curves to save.")
            return
        if self.problem_type == 'regression':
            out_path = os.path.join(out_path, 'regression', f"{self.model_type}_loss_curves.csv")
        elif self.problem_type == 'classification':
            out_path = os.path.join(out_path, 'classification', f"{self.model_type}_loss_curves.csv")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        df.to_csv(out_path, index=False)
        print(f"Loss curves saved to {out_path}")

    def get_feature_importance_analysis(self):
        print("Feature importance analysis not implemented yet.")
        print("Consider implementing SHAP values or permutation importance for better insights.")
        return None


# ## Data Preparation

# In[7]:


master_df = pd.read_parquet(DATA_PATH)


# In[ ]:


master_df.columns


# In[ ]:


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


print(master_df.columns)


# In[ ]:


feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'roll_ret_1d', 'roll_ret_5d', 
    'roll_ret_20d',
    
    'ema_12', 'ema_26', 'ema_50', 'macd_12_26_9', 'macdh_12_26_9',
    'macds_12_26_9', 'rsi_14', 'stochrsik_14_14_3_3', 'stochrsid_14_14_3_3',
    'atrr_14', 'bb_upper', 'bb_middle', 'bb_lower', 'obv',
]

# feature_columns.extend(new_indicator_columns)

all_pipelines = {}
all_results_dfs = {}
all_analyses = {}


# In[ ]:


print(master_df.shape)
master_df = master_df.dropna(subset=feature_columns).sort_values(['ticker','date'])
print(master_df.shape)


# ## Pipeline Execution

# In[ ]:


# print(f"\n{'='*25}\n  RUNNING PIPELINE FOR: GRU\n{'='*25}\n")

# pipeline_BiLSTM = StockPredictionPipeline(
#     df=master_df,
#     feature_columns=feature_columns,
#     model_type='BiGRU',
#     sequence_length=20,
#     problem_type='classification',
#     horizon_steps=1,
#     batch_size=16,
#     learning_rate=0.002559,
#     dropout=0.8,
#     inter_rnn_drop=0.1,
#     lr_factor= 0.8,
#     lr_patience= 7,
#     early_stopping_min_delta=0.000178,
#     early_stopping_patience=15, 
#     num_layers=2,
#     max_epochs=50, 
#     hidden1=256, 
#     hidden2=32,
#     huber_delta=1.35,
#     weight_decay=0.000758,
    
# )

# results_BiLSTM = pipeline_BiLSTM.run_pipeline()

# loss_df = pipeline_BiLSTM.get_loss_curves_df()

# pipeline_BiLSTM.save_loss_curves(str(RESULTS_ROOT))

# if results_BiLSTM is not None and not results_BiLSTM.empty:
#     analysis_GRU = pipeline_BiLSTM.analyze_results()
#     pipeline_BiLSTM.save_results(results_BiLSTM, output_dir=str(RESULTS_ROOT / 'base' / '1H'))
#     all_pipelines["BiLSTM"] = pipeline_BiLSTM
#     all_results_dfs["BiLSTM"] = results_BiLSTM
#     all_analyses["BiLSTM"] = analysis_GRU

#     print("\nDisplaying first 5 rows of BiLSTM results:")
#     print(results_BiLSTM.head())
# else:
#     print(f"\n[FAILED] Pipeline for BiLSTM did not produce any results.")

# del pipeline_BiLSTM


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


# # Hyper Parameter Tuning

# In[ ]:


try:
    import optuna
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna


from pathlib import Path
from datetime import datetime


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

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
    # 'base': ['meta', 'meta_err'],
    'nlp': ['meta_sentiment', 'meta_stance', 'meta_emotion', 'meta_unified_emotion', 'meta_finbert', 'meta_all_nlp', 'meta_err_sentiment', 'meta_err_stance', 'meta_err_emotion', 'meta_err_unified_emotion', 'meta_err_finbert', 'meta_err_all_nlp'],
    # 'sector': ['meta_sector', 'meta_sector_sentiment', 'meta_sector_stance', 'meta_sector_emotion', 'meta_sector_unified_emotion', 'meta_sector_finbert', 'meta_sector_all_nlp', 'meta_err_sector', 'meta_err_sector_sentiment', 'meta_err_sector_stance', 'meta_err_sector_emotion', 'meta_err_sector_unified_emotion', 'meta_err_sector_finbert', 'meta_err_sector_all_nlp'],

}

N_TRIALS = 100
PROBLEM_TYPES = ['classification']

all_optuna_results = {}

for problem_type in PROBLEM_TYPES:
    all_optuna_results[problem_type] = {}
    for feature_group, group_sets in feature_groups.items():
        print(f"=== OPTUNA: {problem_type.upper()} | FEATURE GROUP: {feature_group.upper()} ===")
        
        # Optuna sampler + pruner
        sampler = optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True, group=True, seed=42)
        pruner = optuna.pruners.NopPruner()

        def objective(trial, problem_type=problem_type, group_sets=group_sets):
            params = {
                'problem_type': problem_type,
                'feature_set': trial.suggest_categorical('feature_set', list(group_sets)),
                'model_type': trial.suggest_categorical('model_type', ['LSTM', 'BiLSTM', 'GRU', 'BiGRU']),
                'sequence_length': trial.suggest_int('sequence_length', 5, 20, step=5),
                'horizon_steps': trial.suggest_categorical('horizon_steps', [1]),
                'hidden1': trial.suggest_categorical('hidden1', [64, 128, 256]),
                'hidden2': trial.suggest_categorical('hidden2', [32, 64, 128]),
                'num_layers': trial.suggest_categorical('num_layers', [1, 2]),
                'inter_rnn_drop': trial.suggest_float('inter_rnn_drop', 0.0, 0.4, step=0.1),
                'dropout': trial.suggest_float('dropout', 0.0, 0.8, step=0.1),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True),
                'lr_patience': trial.suggest_categorical('lr_patience', [5, 7, 10]),
                'lr_factor': trial.suggest_categorical('lr_factor', [0.4, 0.8]),
                'early_stopping_patience': trial.suggest_categorical('early_stopping_patience', [10, 15, 20]),
                'max_epochs': trial.suggest_categorical('max_epochs', [20, 30, 50]),
                'huber_delta': trial.suggest_float('huber_delta', 0.1, 2.0),
                'early_stopping_min_delta': trial.suggest_float('early_stopping_min_delta', 0.0, 0.01),
            }

            selected_features = feature_sets[params['feature_set']]
            missing_cols = [c for c in selected_features if c not in master_df.columns]
            if missing_cols:
                print(f"Missing columns for feature_set={params['feature_set']}: {missing_cols}")
                return -1.0

            pipeline = StockPredictionPipeline(
                df=master_df,
                feature_columns=selected_features,
                model_type=params['model_type'],
                sequence_length=params['sequence_length'],
                problem_type=params['problem_type'],
                horizon_steps=params['horizon_steps'],
                hidden1=params['hidden1'],
                hidden2=params['hidden2'],
                num_layers=params['num_layers'],
                inter_rnn_drop=params['inter_rnn_drop'],
                dropout=params['dropout'],
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                lr_patience=params['lr_patience'],
                lr_factor=params['lr_factor'],
                early_stopping_patience=params['early_stopping_patience'],
                max_epochs=params['max_epochs'],
                huber_delta=params['huber_delta'],
                early_stopping_min_delta=params['early_stopping_min_delta'],
                trial=trial
            )

            results_df = pipeline.run_pipeline()
            del pipeline
            empty_cache()

            if results_df is None or results_df.empty:
                print('[DEBUG] results_df empty or None')
                return -1.0

            # Aggregate validation metrics (classification only)
            if problem_type == 'classification':
                val_f1 = results_df['val_f1'].mean() if 'val_f1' in results_df.columns else float('nan')
                if not np.isfinite(val_f1) and 'best_threshold_metric' in results_df.columns:
                    val_f1 = results_df['best_threshold_metric'].mean()
                val_mcc = results_df['val_mcc'].mean() if 'val_mcc' in results_df.columns else float('nan')
                val_precision = results_df['val_precision'].mean() if 'val_precision' in results_df.columns else float('nan')
                val_recall = results_df['val_recall'].mean() if 'val_recall' in results_df.columns else float('nan')
                val_dir_acc = results_df['val_directional_accuracy'].mean() if 'val_directional_accuracy' in results_df.columns else float('nan')

                trial.set_user_attr('val_f1', float(val_f1))
                trial.set_user_attr('val_mcc', float(val_mcc))
                trial.set_user_attr('val_precision', float(val_precision))
                trial.set_user_attr('val_recall', float(val_recall))
                trial.set_user_attr('val_directional_accuracy', float(val_dir_acc))

                score = val_mcc
                if not np.isfinite(score):
                    return -1.0
                return float(score)
            else:
                # minimize MAE -> maximize negative MAE
                if 'mae' not in results_df.columns:
                    return -1.0
                mae = results_df['mae'].mean()
                if not np.isfinite(mae):
                    return -1.0
                return float(-mae)

        # Run study
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(objective, n_trials=N_TRIALS)

        # Collect results
        optuna_results = study.trials_dataframe()
        user_attrs = pd.DataFrame([t.user_attrs for t in study.trials])
        optuna_results = pd.concat([optuna_results, user_attrs], axis=1)
        optuna_results = optuna_results.sort_values('value', ascending=False)
        all_optuna_results[problem_type][feature_group] = optuna_results

        out_dir = RESULTS_ROOT / problem_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'optuna_tuning_{feature_group}_1H.csv'
        optuna_results.to_csv(str(out_path), index=False)
        print(f'Saved Optuna results to {out_path}')

# Display latest results (optional)
all_optuna_results

