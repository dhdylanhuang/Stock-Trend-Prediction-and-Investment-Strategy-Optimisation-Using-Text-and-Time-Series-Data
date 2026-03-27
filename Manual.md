# Manual

## Overview
This repository contains the implementation for the dissertation project:

**Stock Trend Prediction and Investment Strategy Optimisation Using Text and Time‑Series Data**

The project investigates whether combining historical market data, textual signals, reliability‑aware meta‑features, probability calibration, and portfolio allocation can improve both predictive and economic performance in stock forecasting.

The repository supports four main stages:
1. Feature engineering  
2. Predictive benchmarking  
3. Meta‑feature generation  
4. Investment simulation  

## Repository Purpose
The codebase is designed to answer four linked questions:
1. Whether text improves stock prediction relative to price‑only inputs  
2. Whether reliability‑aware meta‑features improve robustness  
3. Whether longer horizons are more useful than next‑day prediction  
4. How predictive outputs translate into trading performance under different allocation rules  

This is an **end‑to‑end research pipeline**, not a collection of standalone models.

## Main Components

### 1. Feature Engineering
Builds the model inputs used throughout the project, including:
- Market variables: `open`, `high`, `low`, `close`, `volume`  
- Technical indicators  
- Sentiment/NLP‑derived features  
- Sector‑level aggregates  
- Meta‑features derived from prior predictions and errors  

### 2. Predictive Benchmarking
Evaluates multiple recurrent architectures across tasks and feature groups.

Tasks:
- Binary classification  
- Regression  
- Ordinal multi‑class classification  

Model families:
- LSTM  
- BiLSTM  
- GRU  
- BiGRU  

### 3. Meta‑Feature Construction
Generates reliability‑aware features from prior model behaviour under a walk‑forward regime:
- Prior predicted returns and probabilities  
- Absolute and squared errors  
- Brier scores and log‑loss  
- Directional correctness  
- Rolling evaluation statistics  

### 4. Investment Simulation
Converts predictive outputs into portfolio decisions and evaluates economic performance using:
- Kelly allocation  
- MPT allocation  
- Hybrid Kelly–MPT allocation  
- Optional early‑exit logic  
- Walk‑forward backtesting  

## Data Requirements
The project expects a merged dataset at **daily ticker level** containing both price‑based and text‑derived information.

### StockNet Dataset (Required)
Download the StockNet dataset by Xu & Cohen and place it in the `data/` folder:

```text
https://github.com/yumoxu/stocknet-dataset
```

Expected location after download:
- `data/stocknet-dataset/`

Market data fields:
- `date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`  

Technical indicators (examples):
- `ema_12`, `ema_26`, `ema_50`, `macd_12_26_9`, `rsi_14`, `bb_upper`, `bb_middle`, `bb_lower`, `obv`  

Textual features (examples):
- `sentiment`, `emotion_*`, `stance_*`, `finbert_*`  

Sector features (examples):
- `sector_open_mean`, `sector_close_mean`, `sector_ret_1d`, `sector_vol_20d`  

Meta‑features (generated later):
- `reg_pred_ret_1d_*`, `cls_prob_up_1d_*`, `reg_abs_err_lag1_*`, `cls_brier_20_*`  

## Recommended Execution Order (Start → Finish)
1. **Prepare master dataset**  
   Goal: produce a clean daily ticker‑level dataset, ensure chronological ordering, handle missing text signals.  
2. **Generate technical and textual features**  
   Run NLP scoring and TA notebooks to create sentiment/emotion/stance/FinBERT + indicators.  
3. **Construct targets**  
   Generate binary, regression, and ordinal targets for required horizons.  
4. **Run benchmarking**  
   Train/evaluate models across feature sets and save results.  
5. **Generate meta‑features**  
   Create reliability‑aware features from out‑of‑sample predictions.  
6. **Run meta‑feature benchmarking**  
   Repeat predictive evaluation with meta‑features included.  
7. **Run investment simulation**  
   Use best configurations for simulated trading with allocation constraints.  

## Environment Setup
**Python**: 3.10+ recommended  

Common libraries:
- `pandas`  
- `numpy`  
- `scikit-learn`  
- `torch`  
- `transformers`  
- `optuna`  
- `matplotlib`  
- `pandas_ta`  
- `pyarrow`  
- `tqdm`  
- `seaborn`  

Install dependencies using your preferred environment manager:

```bash
pip install -r requirements.txt
```

## Hardware Notes
- GPU acceleration supported via CUDA or Apple MPS.  
- Most models are heavy; GPU is recommended for practical runtimes.  

## Outputs
- Intermediate parquet files: `data/dataset/`  
- Benchmark results: `results/benchmarking/`  
- Simulation outputs and logs: `simulation/`  
- Model artifacts: `trained_models*`  

## Troubleshooting
- If NLP models fail to download, verify internet access and Hugging Face cache permissions.  
- If memory errors occur, reduce batch sizes or sequence length.  
- If you see missing columns, re‑run upstream preprocessing or feature notebooks in order.  

## Reproducibility
Most notebooks fix random seeds and avoid data shuffling. Exact reproducibility can still vary across GPU hardware and driver versions.
