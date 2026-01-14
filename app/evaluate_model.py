# app/evaluate_model.py
# ===============================
# Evaluate trained BTC direction model
# ===============================

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ===============================
# CONFIG
# ===============================
DATA_PATH = 'data/processed/features_with_prices_1h.csv'
MODEL_PATH = 'data/processed/xgb_1h_classifier.json'
THRESHOLD = 0.5  # decision threshold

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

print(f"Loaded dataset: {df.shape}")

# ===============================
# REBUILD FEATURES / TARGET
# ===============================
X = df.drop(columns=['target', 'future_return', 'return'])
y = df['target']

# Same temporal split as training
split = int(len(X) * 0.8)
X_test = X.iloc[split:]
y_test = y.iloc[split:]

# ===============================
# LOAD MODEL
# ===============================
model = xgb.Booster()
model.load_model(MODEL_PATH)

dtest = xgb.DMatrix(X_test)

# ===============================
# PREDICTIONS
# ===============================
proba = model.predict(dtest)
pred = (proba > THRESHOLD).astype(int)

# ===============================
# BASIC METRICS
# ===============================
print("\n=== MODEL METRICS ===")
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))

# ===============================
# BASELINE COMPARISON
# ===============================
baseline_pred = np.zeros_like(y_test)
baseline_acc = accuracy_score(y_test, baseline_pred)

print("\n=== BASELINE ===")
print(f"Baseline accuracy (always predict 0): {baseline_acc:.4f}")

# ===============================
# PRICE-LEVEL EVALUATION
# ===============================
results = pd.DataFrame(index=X_test.index)

results['close'] = df.loc[X_test.index, 'close']
results['future_return'] = df.loc[X_test.index, 'future_return']
results['real_target'] = y_test
results['pred_target'] = pred
results['prob_up'] = proba

print("\n=== SAMPLE PREDICTIONS ===")
print(results.head())

# ===============================
# SIGNAL QUALITY CHECK
# ===============================
print("\n=== MEAN FUTURE RETURN BY PREDICTION ===")
print(results.groupby('pred_target')['future_return'].mean())

# ===============================
# PROBABILITY BINNING (VERY IMPORTANT)
# ===============================
results['prob_bin'] = pd.cut(
    results['prob_up'],
    bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0]
)

print("\n=== FUTURE RETURN BY PROBABILITY BIN ===")
print(results.groupby('prob_bin')['future_return'].mean())
