# app/07_train_models.py
import os
import logging
import argparse

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# ===============================
# CONFIG
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = "data/processed/"
MODELS_DIR = "models/"
EXPORTS_DIR = "data/processed/" 

SEED = 42
TEST_SIZE = 0.20

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": -1,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": SEED,
    "n_jobs": -1
}

# ===============================
# LOAD
# ===============================
def load_features(resolution: str = "1h"):
    path = os.path.join(INPUT_DIR, f"dataset_sentiment_target_{resolution}.csv")
    if not os.path.exists(path):
        logging.error(f"No encontrado: {path}")
        return None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")
    logging.info(f"Dataset {resolution} cargado: {df.shape}")
    return df

# ===============================
# PREPARE
# ===============================
def prepare_data(df: pd.DataFrame):
    if "target_up" not in df.columns:
        raise ValueError("Falta 'target_up'")
    y = df["target_up"].astype(int)
    drop_cols = ["target_up", "future_return"] + \
                [c for c in df.columns if c.lower().startswith("future_") or any(p in c.lower() for p in ["open","high","low","close","volume","price"])]
    features = df.drop(columns=drop_cols, errors="ignore")
    constant_cols = [c for c in features.select_dtypes("number").columns if features[c].nunique(dropna=True) <= 1]
    features = features.drop(columns=constant_cols)
    data = pd.concat([features, y], axis=1).dropna()
    features = data[features.columns]
    y = data["target_up"].astype(int)
    split_idx = int(len(features) * (1 - TEST_SIZE))
    X_train = features.iloc[:split_idx]
    X_test  = features.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]
    test_timestamps = features.index[split_idx:]
    logging.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, test_timestamps

# ===============================
# TRAIN & EVALUATE
# ===============================
def train_and_evaluate(X_train, X_test, y_train, y_test, test_timestamps, resolution: str, model_name: str):
    if model_name == "xgb":
        model = XGBClassifier(**XGB_PARAMS)
    elif model_name == "rf":
        model = RandomForestClassifier(**RF_PARAMS)
    else:
        raise ValueError("Modelo no soportado")
    
        os.makedirs(MODELS_DIR, exist_ok=True)

    feature_cols_path = os.path.join(
        MODELS_DIR,
        f"feature_columns_{resolution}.json"
    )

    with open(feature_cols_path, "w") as f:
        json.dump(list(X_train.columns), f)

    logging.info(f"Feature columns guardadas en: {feature_cols_path}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    logging.info(f"[{model_name.upper()} {resolution}] Accuracy: {acc:.4f}")
    logging.info(f"Report:\n{classification_report(y_test, preds)}")

    # Export model
    model_path = os.path.join(MODELS_DIR, f"{model_name}_clf_{resolution}.joblib")
    import joblib
    joblib.dump(model, model_path)
    logging.info(f"Modelo: {model_path}")

    # Export preds 
    preds_df = pd.DataFrame({
        "timestamp": test_timestamps,
        "y_true": y_test,
        "y_pred": preds,
        "proba_up": proba
    })
    preds_path = os.path.join(EXPORTS_DIR, f"predictions_{model_name}_clf_{resolution}.csv")
    preds_df.to_csv(preds_path, index=False)
    logging.info(f"Predicciones: {preds_path}")

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["1h", "4h"], default="4h")
    parser.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    args = parser.parse_args()

    df = load_features(args.resolution)
    if df is not None:
        X_train, X_test, y_train, y_test, test_timestamps = prepare_data(df)
        train_and_evaluate(X_train, X_test, y_train, y_test, test_timestamps, args.resolution, args.model)