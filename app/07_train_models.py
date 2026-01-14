# app/07_train_models.py
# ===============================
# IMPORTS
# ===============================

import os
import logging

import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = 'data/processed/'
OUTPUT_DIR = 'data/processed/'
WINDOWS = ['1h', '4h']

SEED = 42
TEST_SIZE = 0.20  # last 20% for testing (temporal)

# Model params (simple, can be tuned later)
XGB_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.03,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED,
    'n_jobs': -1
}

# ===============================
# LOAD FEATURES
# ===============================
def load_features(window):
    path = os.path.join(INPUT_DIR, f'features_with_prices_{window}.csv')
    if not os.path.exists(path):
        logging.error(f"Features file not found: {path}")
        return None, None, None
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    logging.info(f"Loaded features for {window}: {len(df)} rows")
    return df

# ===============================
# PREPARE DATA
# ===============================
def prepare_data(df):
    # Drop raw OHLCV and any leakage-risk columns if present
    drop_cols = ['open', 'high', 'low', 'close', 'volume']
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    features = df.drop(columns=drop_cols + ['return'], errors='ignore')
    
    # Targets
    y_reg = df['return']
    y_clf = (df['return'] > 0).astype(int)  # 1 = up, 0 = down
    
    # Temporal split (last portion = test)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    
    X_train = features.iloc[:split_idx]
    X_test  = features.iloc[split_idx:]
    
    y_reg_train = y_reg.iloc[:split_idx]
    y_reg_test  = y_reg.iloc[split_idx:]
    
    y_clf_train = y_clf.iloc[:split_idx]
    y_clf_test  = y_clf.iloc[split_idx:]
    
    logging.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    return (X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test)

# ===============================
# TRAIN & EVALUATE REGRESSION
# ===============================
def train_regression(X_train, X_test, y_train, y_test, window):
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    logging.info(f"[Regression {window}] MSE: {mse:.8f}")
    
    # Save
    model_path = os.path.join(OUTPUT_DIR, f'xgb_reg_{window}.json')
    model.get_booster().save_model(model_path)
    logging.info(f"Regression model saved: {model_path}")
    
    return model

# ===============================
# TRAIN & EVALUATE CLASSIFICATION
# ===============================
def train_classification(X_train, X_test, y_train, y_test, window):
    model = XGBClassifier(**XGB_PARAMS, objective='binary:logistic')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    logging.info(f"[Classification {window}] Accuracy: {acc:.4f}")
    logging.info(f"Classification report:\n{classification_report(y_test, preds)}")
    
    # Save
    model_path = os.path.join(OUTPUT_DIR, f'xgb_clf_{window}.json')
    model.get_booster().save_model(model_path)
    logging.info(f"Classification model saved: {model_path}")
    
    return model

# ===============================
# MAIN TRAINING FLOW
# ===============================
def train_for_window(window):
    df = load_features(window)
    if df is None:
        return
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = prepare_data(df)
    
    # Train both models
    train_regression(X_train, X_test, y_reg_train, y_reg_test, window)
    train_classification(X_train, X_test, y_clf_train, y_clf_test, window)

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    for w in WINDOWS:
        logging.info(f"\n=== Training models for {w} ===")
        train_for_window(w)
    logging.info("All model training completed.")