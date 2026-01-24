# Sentiment Analysis Pipeline for BTC Price Prediction

## Overview
This branch implements the sentiment analysis component of a hybrid BTC price prediction system for 4Geeks Academy. The goal: predict up/down movements in 1H/4H resolutions using ~30% sentiment features + ~70% time-series (ARIMA/SARIMA). 
Final integration: dynamic preds in Streamlit app (historical lookup or live fetch/compute/predict).

Focus: NLP pipeline on Bitcoin news for sentiment scores, aggregated features (rolls/lags/shocks/div/ratio + Fear&Greed), merged with prices for training RF/XGB classifiers. Supports live mode with real-time news/F&G.

## Pipeline Stages
1. **Fetch News (`fetch_news.py`)**: Downloads news from NewsAPI by axis (BTC, MACRO, TECH) using keywords. Supports date ranges or live (last 24H). Outputs: raw CSV with timestamps, titles, descriptions, sources.

2. **Compute Sentiment (`compute_sentiment.py`)**: Ensemble VADER + FinBERT on title+description. Batched processing, truncation for long texts. Outputs: interim CSV with vader_score, finbert_score, ensemble sentiment_score (-1 to 1).

3. **Aggregate Features (`aggregate_features.py`)**: Aggregates to 1H/4H freq per axis. Stats: mean/std/count. Engineers: rolling mean/std (7 periods), shocks (±2 std), lags(1), momentum(diff), divergences (BTC-MACRO), ratios (BTC/MACRO count). Integrates Fear&Greed (CSV or API via download_latest_fng.py, ffill/default 50). Outputs: processed CSVs (aggregated_1h/4h.csv).

4. **Integrate Features & Target (`integrate_features_target.py`)**: Merges aggregated features with BTC prices (btcusd-1h/4h.csv). Computes target_up (binary: future_return >0) and future_return (shift -1/-4). Drops close to avoid leakage. Outputs: dataset_sentiment_target_1h/4h.csv for training.

5. **Train Models (`train_models.py`)**: Loads integrated dataset, splits (80/20 chronological), trains RF/XGB classifiers (params: n_est=200, etc.). Evaluates: acc, report, confusion. Exports: models/rf_clf_1h.joblib, etc.; predictions CSVs.

6. **Dynamic Prediction (`dynamic_predict.py`)**: Core for Streamlit. Parses input (ts, res, mode=auto, window=24, model=xgb/rf). Auto: hist if in dataset (nearest row), else live (fetch news → sentiment → aggregate + F&G → predict). Returns dict: prediction (up/down), confidence, proba_up, mode_used, msg/error.

## Exploratory Data Analysis (EDAs)
- **eda_sentiments.ipynb**: Justifies sentiment pipeline. Assesses data quality (temporal consistency, zero-handling), feature stats (episodic/non-stationary), weak corrs with returns but regime effects (high volume/divergence). Concludes: sentiment as contextual modulator.
- **eda_time_series.ipynb**: (Colab with Carlos) Decomposes prices, tests stationarity (ADF), ACF/PACF for ARIMA/SARIMA params. Supports time-series baseline.

## Models & Outputs
- Classifiers: RF (max_depth=10), XGB (lr=0.03, depth=5). Binary: target_up.
- Data: raw (news/prices/F&G), interim (sentiment/aggregated), processed (datasets/preds).
- Models: .joblib in models/.

## Integration & Usage
- Streamlit (`streamlit_app.py`): User selects ts/res → dynamic_predict → displays preds/charts/news. Modes: hist/live.
- Run: `python fetch_news.py` → `compute_sentiment.py` → etc. For live: dynamic_predict handles flow.
- Deps: requirements.txt (pandas, sklearn, xgboost, transformers, vaderSentiment, newsapi, etc.).

## Conclusions
Implemented full NLP sentiment pipeline, feature engineering, training, and dynamic inference. Achieves contextual signals for BTC preds, ready for fusion (~30% weight) and app deployment. Weaknesses: weak sentiment corrs (expected, per EDA); strengths: live-capable, modular.