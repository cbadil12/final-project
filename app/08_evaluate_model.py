# app/08_evaluate_model.py
# ===============================
# IMPORTS
# ===============================

import os
import logging

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES_PATH = 'data/processed/features_with_prices_1h.csv'  # Cambiar a 4h si quieres
MODEL_PATH = 'data/processed/xgb_clf_1h.json'
THRESHOLD = 0.5          # Default decision threshold
PROBA_BINS = [0, 0.4, 0.5, 0.6, 0.7, 1.0]

TEST_SIZE = 0.20         # Must match training split

# ===============================
# LOAD DATA & MODEL
# ===============================
def load_data_and_model():
    if not os.path.exists(FEATURES_PATH):
        logging.error(f"Features not found: {FEATURES_PATH}")
        return None, None, None
    
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    # Features + targets
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'return']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    
    y_true = (df['return'] > 0).astype(int)  # 1 = up, 0 = down
    future_returns = df['return']
    closes = df['close']
    
    # Temporal split (same as training)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    X_test = X.iloc[split_idx:]
    y_test = y_true.iloc[split_idx:]
    future_ret_test = future_returns.iloc[split_idx:]
    close_test = closes.iloc[split_idx:]
    
    logging.info(f"Test set size: {len(X_test)} rows")
    
    # Load model
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    
    dtest = xgb.DMatrix(X_test)
    
    return dtest, y_test, future_ret_test, close_test, model

# ===============================
# EVALUATE & FINANCIAL SIMULATION
# ===============================
def evaluate():
    dtest, y_test, future_ret, close, model = load_data_and_model()
    if dtest is None:
        return
    
    # Predictions
    proba = model.predict(dtest)
    pred = (proba > THRESHOLD).astype(int)
    
    # Classification metrics
    acc = accuracy_score(y_test, pred)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info(f"\nClassification Report:\n{classification_report(y_test, pred)}")
    logging.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, pred)}")
    
    # Prepare results DF
    results = pd.DataFrame(index=y_test.index)
    results['close'] = close
    results['future_return'] = future_ret
    results['true'] = y_test
    results['pred'] = pred
    results['prob_up'] = proba
    
    # Simple strategy: long when pred=1, flat otherwise
    results['signal'] = results['pred']
    results['strategy_return'] = results['signal'].shift(1) * results['future_return']  # No look-ahead
    
    # Financial metrics (no transaction costs yet)
    strat_ret = results['strategy_return'].fillna(0)
    cum_ret = (1 + strat_ret).cumprod()
    total_return = cum_ret.iloc[-1] - 1
    n_trades = (results['signal'].diff() != 0).sum() / 2  # approx changes
    win_rate = (strat_ret > 0).mean() if len(strat_ret[strat_ret != 0]) > 0 else 0
    
    logging.info("\n=== SIMPLE STRATEGY (no costs) ===")
    logging.info(f"Total compounded return: {total_return:.4%}")
    logging.info(f"Approx number of trades: {int(n_trades)}")
    logging.info(f"Win rate: {win_rate:.1%}")
    
    # Sharpe (rough annualization for 1h: ~8760 periods/year)
    if strat_ret.std() > 0:
        sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(8760)
        logging.info(f"Sharpe ratio (annualized approx): {sharpe:.2f}")
    
    # Probability binning
    results['prob_bin'] = pd.cut(results['prob_up'], bins=PROBA_BINS, include_lowest=True)
    logging.info("\n=== Mean future return by probability bin ===")
    print(results.groupby('prob_bin')['future_return'].mean())
    
    # Baseline: always long
    always_long_ret = future_ret.mean()
    logging.info(f"\nBaseline (always long): {always_long_ret:.6f} avg return per period")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    evaluate()