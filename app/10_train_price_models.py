# app/10_train_price_models.py
import os
import json
import logging
import argparse

import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEED = 42
TEST_SIZE = 0.20

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": SEED,
    "n_jobs": -1,
    "class_weight": "balanced",
}

DATASET_1H = "data/processed/dataset_price_target_1h.csv"
DATASET_4H = "data/processed/dataset_price_target_4h.csv"
DATASET_24H = "data/processed/dataset_price_target_24h.csv"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.dropna().sort_index()
    return df


def prepare_xy(df: pd.DataFrame):
    y = df["target_up"].astype(int)

    drop_cols = ["target_up", "future_return"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # drop raw OHLC to avoid the model learning trivial scaling;
    # keep close_lags etc already in features
    raw_cols = [c for c in ["open", "high", "low", "close"] if c in X.columns]
    X = X.drop(columns=raw_cols, errors="ignore")

    # time split
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    ts_test = X.index[split:]

    return X_train, X_test, y_train, y_test, ts_test


def train(model_name: str, resolution: str):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # --- dataset path for resolution
    if resolution == "1h":
        ds_path = DATASET_1H
    elif resolution == "4h":
        ds_path = DATASET_4H
    elif resolution == "24h":
        ds_path = DATASET_24H
    else:
        raise ValueError("resolution must be one of: 1h, 4h, 24h")

    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset not found: {ds_path}")

    df = load_dataset(ds_path)
    X_train, X_test, y_train, y_test, ts_test = prepare_xy(df)

    if model_name == "xgb_price":
        model = XGBClassifier(**XGB_PARAMS)
    elif model_name == "rf_price":
        model = RandomForestClassifier(**RF_PARAMS)
    else:
        raise ValueError("model_name must be xgb_price or rf_price")

    # save schema (names + order) for this model
    schema_path = os.path.join("models", f"feature_columns_{model_name}_{resolution}.json")
    with open(schema_path, "w") as f:
        json.dump(list(X_train.columns), f)
    logging.info(f"Schema saved: {schema_path}")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    logging.info(f"[{model_name} {resolution}] Accuracy: {acc:.4f}")
    logging.info(f"Report:\n{classification_report(y_test, preds)}")

    model_path = os.path.join("models", f"{model_name}_{resolution}.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Model saved: {model_path}")

    out_preds = pd.DataFrame(
        {"timestamp": ts_test, "y_true": y_test.values, "y_pred": preds, "proba_up": proba}
    )
    preds_path = os.path.join("data/processed", f"predictions_{model_name}_{resolution}.csv")
    out_preds.to_csv(preds_path, index=False)
    logging.info(f"Predictions saved: {preds_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["1h", "4h", "24h"], default="1h")
    parser.add_argument("--model", choices=["xgb_price", "rf_price"], default="xgb_price")
    args = parser.parse_args()

    train(args.model, args.resolution)
