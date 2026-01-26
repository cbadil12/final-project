# app/run_model.py

# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import logging

# 2. Third-party libraries
import pandas as pd
import numpy as np
import joblib
import json

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_project_root() -> str:
    cwd = os.getcwd()
    base = os.path.basename(cwd)
    if base in {"app", "src", "notebooks"}:
        return os.path.abspath(os.path.join(cwd, ".."))
    return cwd

PROJECT_ROOT = get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_EXT = '.joblib'
THRESHOLD = 0.5  # For binary prediction (up if proba_up > threshold)
DEFAULT_MODEL = 'xgb_clf'  # Default if not specified
DEFAULT_RESOLUTION = '1h'

# Supported models and resolutions (from your tree)
SUPPORTED = {
    "sentiment": {
        "models": ["rf_clf", "xgb_clf"],
        "resolutions": ["1h", "4h", "24h"],
    },
    "price": {
        "models": ["rf_price", "xgb_price"],
        "resolutions": ["1h", "4h", "24h"],
    },
}

def load_feature_columns(task: str, model_name: str, resolution: str):
    # sentiment: feature_columns_{resolution}.json  (legacy)
    # price:     feature_columns_{model_name}_{resolution}.json (nuevo)
    if task == "sentiment":
        path = os.path.join(MODEL_DIR, f"feature_columns_{resolution}.json")
    elif task == "price":
        path = os.path.join(MODEL_DIR, f"feature_columns_{model_name}_{resolution}.json")
    else:
        raise ValueError("task must be 'sentiment' or 'price'")

    if not os.path.exists(path):
        logging.warning(f"Feature columns file not found: {path}")
        return None

    with open(path, "r") as f:
        cols = json.load(f)
    return cols if isinstance(cols, list) and cols else None


def align_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    out = df.copy()
    # add missing
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0
    # drop extra
    extra = [c for c in out.columns if c not in feature_cols]
    if extra:
        out = out.drop(columns=extra, errors="ignore")
    # reorder
    return out[feature_cols]


# ===============================
# MODEL LOADING
# ===============================
from functools import lru_cache

@lru_cache(maxsize=16)
def load_model(task: str = "sentiment", resolution: str = DEFAULT_RESOLUTION, model_name: str = DEFAULT_MODEL):
    if task not in SUPPORTED:
        raise ValueError(f"Unsupported task: {task}. Supported: {list(SUPPORTED.keys())}")

    if resolution not in SUPPORTED[task]["resolutions"]:
        raise ValueError(
            f"Unsupported resolution for {task}: {resolution}. Supported: {SUPPORTED[task]['resolutions']}"
        )

    if model_name not in SUPPORTED[task]["models"]:
        raise ValueError(
            f"Unsupported model for {task}: {model_name}. Supported: {SUPPORTED[task]['models']}"
        )

    model_filename = f"{model_name}_{resolution}{MODEL_EXT}"
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    logging.info(f"✅ Model loaded: {model_filename}")
    return model

# ===============================
# PREDICTION FUNCTION
# ===============================
def run_prediction(
    features_df: pd.DataFrame,
    target_ts: pd.Timestamp,
    model,
    resolution: str,
    task: str,
    model_name: str,
    drop_cols: list = None
) -> dict:
    """
    Runs prediction on the features DF for the target_ts.
    Selects nearest row, aligns features with training schema, predicts proba.

    Args:
        features_df: DF with datetime index and feature columns
        target_ts: Timestamp to predict for (UTC)
        model: Loaded model object
        resolution: '1h' | '4h' | '24h'
        task: 'sentiment' | 'price'
        model_name: model identifier (e.g. rf_price, xgb_clf)
        drop_cols: Optional list of non-feature columns to drop

    Returns:
        dict: {'prediction': 0/1, 'proba_up': float, 'confidence': float}
    """

    # --- empty safety
    if features_df is None or features_df.empty:
        logging.warning("Empty features DF - returning neutral fallback")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}

    # --- ensure UTC timestamp
    target_ts = pd.to_datetime(target_ts, utc=True)

    # --- ensure DatetimeIndex UTC
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index, utc=True, errors="coerce")
    elif features_df.index.tz is None:
        features_df.index = features_df.index.tz_localize("UTC")

    features_df = features_df.sort_index()

    # --- find nearest row
    try:
        pos = features_df.index.get_indexer([target_ts], method="nearest")[0]
        row = features_df.iloc[[pos]].copy()
    except Exception as e:
        logging.error(f"Failed to select row for {target_ts}: {e}")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}

    # --- drop non-feature cols
    if drop_cols:
        row = row.drop(columns=[c for c in drop_cols if c in row.columns], errors="ignore")

    # --- load schema used in training
    feature_cols = load_feature_columns(task, model_name, resolution)
    if feature_cols:
        row = align_features(row, feature_cols)

    # --- final NaN safety
    row = row.fillna(0.0)

    # --- predict
    try:
        X = row.values.astype(np.float32)
        proba = model.predict_proba(X)[0]
        proba_up = float(proba[1])
        pred = 1 if proba_up >= THRESHOLD else 0
        confidence = max(proba_up, 1 - proba_up)

        logging.info(
            f"[{task.upper()} | {model_name} | {resolution}] "
            f"ts={target_ts} pred={pred} proba_up={proba_up:.3f}"
        )

        return {
            "prediction": pred,
            "proba_up": proba_up,
            "confidence": confidence,
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}


# ===============================
# ENTRY POINT (for standalone test)
# ===============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "price"], default="sentiment")
    parser.add_argument("--resolution", choices=["1h", "4h", "24h"], default="1h")
    parser.add_argument("--model", default=None)  # si None -> usa DEFAULT_MODEL según task
    parser.add_argument("--input", default=None)  # ruta al features CSV
    parser.add_argument("--dump_csv", action="store_true")  # guarda salida a data/processed/
    parser.add_argument("--all", action="store_true")  # corre todos los modelos/resoluciones del task
    args = parser.parse_args()

    # defaults por task
    default_model_by_task = {
        "sentiment": "xgb_clf",
        "price": "rf_price",
    }

    # input por defecto (solo para test rápido)
    default_input_by_task = {
        "sentiment": {
            "1h": "data/interim/aggregated_1h.csv",
            "4h": "data/interim/aggregated_4h.csv",
            "24h": "data/interim/aggregated_24h.csv",
        },
        "price": {
            "1h": "data/interim/price_features_1h.csv",
            "4h": "data/interim/price_features_4h.csv",
            "24h": "data/interim/price_features_24h.csv",
        },
    }

    task = args.task
    if task not in SUPPORTED:
        raise ValueError(f"Unsupported task: {task}")

    # función helper para ejecutar 1 combo
    def _run_one(task: str, resolution: str, model_name: str, input_path: str):
        if not os.path.exists(input_path):
            logging.warning(f"Test skipped (missing input): {input_path}")
            return

        df_features = pd.read_csv(input_path, index_col=0)

        model = load_model(task=task, resolution=resolution, model_name=model_name)
        last_ts = pd.to_datetime(df_features.index[-1])

        result = run_prediction(
            df_features,
            last_ts,
            model,
            resolution=resolution,
            task=task,
            model_name=model_name,
        )

        print("\n" + "=" * 40)
        print(f"✅ TASK={task} | MODEL={model_name} | RES={resolution}")
        print(f"TS: {last_ts}")
        print(f"DIRECTION: {'UP 📈' if result['prediction'] == 1 else 'DOWN 📉'}")
        print(f"PROBA_UP: {result['proba_up']:.4f}")
        print(f"CONFIDENCE: {result['confidence']:.2%}")
        print("=" * 40 + "\n")

        if args.dump_csv:
            os.makedirs("data/processed", exist_ok=True)
            out_path = os.path.join(
                "data/processed",
                f"standalone_pred_{task}_{model_name}_{resolution}.csv"
            )
            pd.DataFrame([{
                "timestamp": last_ts,
                "task": task,
                "model": model_name,
                "resolution": resolution,
                "prediction": result["prediction"],
                "proba_up": result["proba_up"],
                "confidence": result["confidence"],
            }]).to_csv(out_path, index=False)
            logging.info(f"Saved standalone CSV: {out_path}")

    # --- ALL MODE
    if args.all:
        for res in SUPPORTED[task]["resolutions"]:
            for m in SUPPORTED[task]["models"]:
                input_path = default_input_by_task[task].get(res)
                if input_path:
                    _run_one(task, res, m, input_path)
        raise SystemExit(0)

    # --- single run
    resolution = args.resolution
    model_name = args.model or default_model_by_task[task]
    input_path = args.input or default_input_by_task[task].get(resolution)

    if input_path is None:
        raise ValueError("No input path provided. Use --input <path_to_features_csv>")

    _run_one(task, resolution, model_name, input_path)
