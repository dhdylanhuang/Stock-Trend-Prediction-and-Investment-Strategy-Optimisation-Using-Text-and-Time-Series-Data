# Stock Trend Prediction and Investment Strategy Optimisation (Text + Time Series)

This project builds an end‑to‑end pipeline for stock trend prediction using price/technical indicators and NLP features from financial tweets. It includes data preprocessing, feature engineering (sentiment/emotion/stance/FinBERT), model benchmarking, and investment simulation.

**Status**: active research/engineering workspace (Jupyter notebooks).

**Repository Layout**
- `data-pre-processing/` data ingestion and cleaning notebooks.
- `feat-engineering/` NLP feature generation, technical indicators, sector/meta features.
- `benchmarking/` model training and evaluation notebooks (classification/regression/multiclass).
- `simulation/` investment simulation notebooks.
- `data/` datasets and intermediate parquet outputs.
- `results/` saved benchmarking outputs.
- `documentation/` dissertation and supporting material.

**Key Notebooks**
- `data-pre-processing/Data_PreProcessing_1.ipynb` tweet parsing/cleaning and parquet output.
- `data-pre-processing/Data_PreProcessing_2.ipynb` stock table cleaning (tab‑separated) to parquet.
- `data-pre-processing/Data_PreProcessing_3.ipynb` additional tweet normalization/cleaning.
- `feat-engineering/NLP_1_Sentiment_Scoring.ipynb` tweet sentiment scoring.
- `feat-engineering/NLP_2_0_Emotion_Scoring.ipynb` raw emotion scoring + percentile features.
- `feat-engineering/NLP_2_1_Emotion_Engineering.ipynb` unified emotion features.
- `feat-engineering/NLP_3_Stance_Scoring.ipynb` stance label/score.
- `feat-engineering/NLP_4_FinBert_Sentiment.ipynb` FinBERT sentiment features.
- `feat-engineering/Technical_Indicators.ipynb` price TA indicators + NLP merge.
- `feat-engineering/Sector_Features.ipynb` sector‑level features.
- `feat-engineering/Meta_Features.ipynb` meta‑model signals and reliability features.
- `benchmarking/Benchmarking.ipynb` main training/evaluation pipeline.
- `simulation/Invesment_Simulation_System.ipynb` and `simulation/MultiClass_Invesment_Simulation_System.ipynb`.

**Data Expectations**
- StockNet dataset under `data/stocknet-dataset/`.
- Parquet outputs written to `data/dataset/`.

**Benchmark Split (Fixed Dates)**
Used in all benchmarking notebooks:
- Train: `2014-01-01` to `2015-08-01`
- Validation: `2015-08-01` to `2015-10-01`
- Test: `2015-10-01` to `2016-01-01`

**Quickstart**
1. Place raw dataset in `data/stocknet-dataset/`.
2. Run preprocessing notebooks in `data-pre-processing/`.
3. Run NLP and TA notebooks in `feat-engineering/` in order.
4. Run `benchmarking/Benchmarking.ipynb` or the parallel/colab variants.
5. Run simulation notebooks in `simulation/`.

**Environment Notes**
- Python 3.10+ recommended.
- Common dependencies: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `optuna`, `pandas_ta`, `pyarrow`, `tqdm`, `matplotlib`, `seaborn`.
- GPU support is enabled in the benchmarking notebooks (CUDA or Apple MPS).

**Outputs**
- Intermediate parquet files under `data/dataset/`.
- Benchmark results under `results/benchmarking/`.