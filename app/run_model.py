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
SUPPORTED_MODELS = ['rf_clf', 'xgb_clf']  # Add more if needed
SUPPORTED_RESOLUTIONS = ['1h', '4h']  # Align with app intervals

def load_feature_columns(resolution: str):
    path = os.path.join(MODEL_DIR, f"feature_columns_{resolution}.json")
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
def load_model(resolution: str = DEFAULT_RESOLUTION, model_name: str = DEFAULT_MODEL):
    """
    Loads the pre-trained model from disk.
    Args:
        resolution: '1h' or '4h'
        model_name: 'xgb_clf', 'rf_clf', etc.
    Returns:
        Loaded model object
    Raises:
        ValueError if unsupported, FileNotFoundError if missing
    """
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}. Supported: {SUPPORTED_RESOLUTIONS}")
    
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {SUPPORTED_MODELS}")
    
    model_filename = f"{model_name}_{resolution}{MODEL_EXT}"
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    try:
        model = joblib.load(model_path)
        logging.info(f"✅ Model loaded: {model_filename}")
        return model
    except Exception as e:
        logging.error(f"Error loading model {model_filename}: {e}")
        raise

# ===============================
# PREDICTION FUNCTION
# ===============================
def run_prediction(
    features_df: pd.DataFrame,
    target_ts: pd.Timestamp,
    model,
    resolution: str,
    drop_cols: list = None  # Optional: cols to drop if not features (e.g., ['timestamp'])
) -> dict:
    """
    Runs prediction on the features DF for the target_ts.
    Selects nearest/exact row, prepares input, predicts proba.
    
    Args:
        features_df: DF from aggregate_features (index: timestamp, cols: features)
        target_ts: Timestamp to predict for (UTC)
        model: Loaded model object
        drop_cols: List of non-feature cols to drop (default: none)
    
    Returns:
        dict: {'prediction': 0/1 (down/up), 'proba_up': float, 'confidence': float}
    """
    if features_df.empty:
        logging.warning("Empty features DF - returning neutral fallback")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}
    
    # Ensure target_ts is UTC Timestamp
    target_ts = pd.to_datetime(target_ts, utc=True)

    # Ensure index is UTC DatetimeIndex
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index, utc=True, errors="coerce")
    else:
        # force UTC if naive
        if features_df.index.tz is None:
            features_df.index = features_df.index.tz_localize("UTC")

    features_df = features_df.dropna(axis=0, how="any", subset=[]).sort_index()

    # Nearest row (robust)
    pos = features_df.index.get_indexer([target_ts], method="nearest")[0]
    row = features_df.iloc[[pos]].copy()

    
    # Drop non-feature cols if specified
    if drop_cols:
        row = row.drop(columns=[c for c in drop_cols if c in row.columns], errors='ignore')
    
    # Handle NaNs (simple fill 0 for now - adjust if needed)
    row = row.fillna(0)
    feature_cols = load_feature_columns(resolution)
    if feature_cols:
        row = align_features(row, feature_cols)
    
    # To np.array (1 sample)
    X = row.values.astype(np.float32)
    
    try:
        proba = model.predict_proba(X)[0]  # Assumes binary classifier [down, up]
        proba_up = proba[1]  # Probability of class 1 (up)
        pred = 1 if proba_up > THRESHOLD else 0
        confidence = max(proba_up, 1 - proba_up)  # Higher prob wins
        
        logging.info(f"Prediction for {target_ts}: pred={pred}, proba_up={proba_up:.3f}, conf={confidence:.3f}")
        
        return {
            'prediction': pred,
            'proba_up': proba_up,
            'confidence': confidence
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {'prediction': 0, 'proba_up': 0.5, 'confidence': 0.5}

# ===============================
# ENTRY POINT (for standalone test)
# ===============================
if __name__ == "__main__":
    # Path to real data generated in the previous step
    TEST_INPUT = 'data/interim/aggregated_1h.csv'
    
    if os.path.exists(TEST_INPUT):
        logging.info("--- RUNNING REAL-DATA PREDICTION TEST ---")
        
        # Load the aggregated features
        df_features = pd.read_csv(TEST_INPUT, index_col=0)
        
        try:
            # 1. Load the specific model
            model = load_model(resolution='1h', model_name='xgb_clf')
            
            # 2. Get the most recent timestamp from our data
            last_ts = pd.to_datetime(df_features.index[-1])
            
            # 3. Execute prediction
            result = run_prediction(df_features, last_ts, model, resolution="1h")
            
            # Final Console Output
            print("\n" + "="*30)
            print(f"🚀 PREDICTION FOR: {last_ts}")
            print(f"DIRECTION: {'UP 📈' if result['prediction'] == 1 else 'DOWN 📉'}")
            print(f"PROBABILITY: {result['proba_up']:.4f}")
            print(f"CONFIDENCE: {result['confidence']:.2%}")
            print("="*30 + "\n")
            
        except Exception as e:
            logging.error(f"Test failed during execution: {e}")
    else:
        logging.warning(f"Test skipped: Input file not found at {TEST_INPUT}")
        logging.info("Please run 'aggregate_features.py' first to generate test data.")