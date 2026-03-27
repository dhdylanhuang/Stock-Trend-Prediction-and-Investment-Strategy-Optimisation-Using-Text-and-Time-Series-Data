# Stock Trend Prediction and Investment Strategy Optimisation  
**Text + Time‑Series Signals | Dissertation‑Ready Repository**

This repository presents an end‑to‑end research pipeline for stock trend prediction that fuses price/technical indicators with NLP signals derived from financial tweets. It supports data preprocessing, feature engineering (including sector and meta‑features), model benchmarking (binary, regression, and multi‑class/ordinal), and investment simulation with an optional early‑exit filter under realistic constraints.

## Project Overview
**Goal**: Predict short‑/-medium horizon price direction/returns and evaluate decision‑making performance using a constrained trading simulation.  
**Core idea**: Combine market micro‑signals (technical indicators) with textual sentiment/emotion/stance information to improve predictive signal quality.

### Key Contributions
1. A structured multi‑stage pipeline from raw data to portfolio simulation.  
2. Multiple NLP feature streams (sentiment, emotion, stance, FinBERT) aligned to daily ticker data.  
3. Comparative benchmarking across feature sets and model families.  
4. Simulation framework enforcing allocation limits, diversification, and probabilistic sizing.

## Repository Layout
- `data-pre-processing/` data ingestion and cleaning notebooks.  
- `feat-engineering/` NLP feature generation, technical indicators, sector/meta features.  
- `meta-features/` auxiliary meta‑features and early‑exit signals notebooks.  
- `benchmarking/` model training and evaluation notebooks (binary, regression, multiclass).  
- `simulation/` investment simulation notebooks.  
- `data/` raw datasets and intermediate parquet outputs.  
- `results/` saved benchmarking outputs.  
- `documentation/` dissertation and supporting material.  

## Pipeline Summary
**Stages**
1. **Preprocessing**: parse StockNet price/tweet data and output cleaned parquet files.  
2. **Feature Engineering**: NLP features, technical indicators, sector features, and meta‑features.  
3. **Benchmarking**: train/evaluate sequence models on fixed time splits.  
4. **Meta‑Features & Early‑Exit**: generate model‑reliability signals and optional early‑exit predictions for simulation.  
5. **Simulation**: portfolio construction using calibrated predictions and constrained risk rules.

## Pipeline Run Order (Start → Finish)
**Start here**: raw StockNet data in `data/stocknet-dataset/`  
**End here**: benchmark results in `results/benchmarking/` and simulation outputs/logs in `simulation/` and `trained_models*`

**Recommended execution path**
1. **Preprocess**  
   Run: `data-pre-processing/Data_PreProcessing_1.ipynb` → `Data_PreProcessing_2.ipynb` → `Data_PreProcessing_3.ipynb`  
   Output: cleaned parquet files in `data/dataset/`
2. **NLP Features**  
   Run: `NLP_1_Sentiment_Scoring.ipynb` → `NLP_2_0_Emotion_Scoring.ipynb` → `NLP_2_1_Emotion_Engineering.ipynb` →  
   `NLP_3_Stance_Scoring.ipynb` → `NLP_4_FinBert_Sentiment.ipynb`
3. **Technical + Sector Features**  
   Run: `Technical_Indicators.ipynb` → `Sector_Features.ipynb`
4. **Meta‑Features & Early‑Exit**  
   Run: `feat-engineering/Meta_Features.ipynb` → `meta-features/Early_Exit.ipynb`
5. **Benchmarking**  
   Run: `benchmarking/Benchmarking.ipynb` and/or `benchmarking/MultiClass_Benchmarking.ipynb`, can be done on parquet with or without meta-features.
6. **Simulation (End of Pipeline)**  
   Run: `simulation/Invesment_Simulation_System.ipynb` and/or  
   `simulation/MultiClass_Invesment_Simulation_System.ipynb`


## System Architecture
![Pipeline Overview](documentation/architecturediagram.svg)

## Notebook Guide (Primary)
**Preprocessing**
- `data-pre-processing/Data_PreProcessing_1.ipynb` tweet parsing/cleaning and parquet output.  
- `data-pre-processing/Data_PreProcessing_2.ipynb` stock table cleaning (tab‑separated) to parquet.  
- `data-pre-processing/Data_PreProcessing_3.ipynb` tweet normalization/cleaning (non‑merged).  

**NLP Features**
- `feat-engineering/NLP_1_Sentiment_Scoring.ipynb` sentiment scoring (1–5).  
- `feat-engineering/NLP_2_0_Emotion_Scoring.ipynb` emotion scores + percentiles.  
- `feat-engineering/NLP_2_1_Emotion_Engineering.ipynb` unified emotion features.  
- `feat-engineering/NLP_3_Stance_Scoring.ipynb` stance label/score.  
- `feat-engineering/NLP_4_FinBert_Sentiment.ipynb` FinBERT sentiment features.  

**Technical + Sector Features**
- `feat-engineering/Technical_Indicators.ipynb` TA indicators + NLP merge.  
- `feat-engineering/Sector_Features.ipynb` sector‑level aggregates and indicators.  

**Meta‑Features + Early Exit**
- `feat-engineering/Meta_Features.ipynb` meta‑model signals and reliability features.  
- `meta-features/Early_Exit.ipynb` early‑exit signals for filtering simulation trades.  

**Benchmarking**
- `benchmarking/Benchmarking.ipynb` binary classification & regression pipeline.  
- `benchmarking/MultiClass_Benchmarking.ipynb` multi‑class/ordinal pipeline.  
- `benchmarking/*_GPU.ipynb` GPU‑optimized variants for cluster runs.  

**Simulation**
- `simulation/Invesment_Simulation_System.ipynb` binary simulation engine.  
- `simulation/MultiClass_Invesment_Simulation_System.ipynb` ordinal simulation engine.  
- `meta-features/Early_Exit.ipynb` early‑exit features for simulation filtering.  

## Data Expectations
- StockNet dataset under `data/stocknet-dataset/`.  
- Intermediate parquet outputs written to `data/dataset/`.

## Benchmark Split (Fixed Dates)
Used across benchmarking notebooks:
- Train: **2014‑01‑01** to **2015‑08‑01**  
- Validation: **2015‑08‑01** to **2015‑10‑01**  
- Test: **2015‑10‑01** to **2016‑01‑01**  

## Simulation Notes
The simulation uses walk‑forward training, per‑ticker models, calibrated probabilities, and constrained allocation. It enforces:
- total capital utilization caps,  
- per‑ticker caps,  
- optional sector diversification,  
- drawdown‑based throttling,  
- fractional Kelly sizing.

## Quickstart (Recommended Order)
1. Place raw dataset in `data/stocknet-dataset/`.  
2. Run preprocessing notebooks in `data-pre-processing/`.  
3. Run NLP + TA notebooks in `feat-engineering/` in the order listed above.  
4. Run meta‑features and early‑exit notebooks (`feat-engineering/Meta_Features.ipynb`, `meta-features/Early_Exit.ipynb`).  
5. Run benchmarking notebooks in `benchmarking/`.  
6. Run simulation notebooks in `simulation/`.  

## Environment Notes
**Python**: 3.10+  
**Common dependencies**: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `optuna`, `pandas_ta`, `pyarrow`, `tqdm`, `matplotlib`, `seaborn`  
**GPU**: supported via CUDA or Apple MPS in the benchmarking/simulation notebooks.  

## Outputs
- Intermediate parquet files: `data/dataset/`  
- Benchmark results: `results/benchmarking/`  
- Simulation logs and artifacts: `simulation/` and `trained_models*`  

## Reproducibility
Most notebooks set fixed random seeds and use deterministic options where possible. Exact reproducibility may still vary across GPU devices and driver versions.

## Limitations
- StockNet coverage and tweet noise may introduce data sparsity or bias.  
- Results are sensitive to feature selection and horizon choice.  
- Simulated trading does not include all real‑world frictions unless explicitly modeled in notebooks.

## License and Usage
For academic use only.
