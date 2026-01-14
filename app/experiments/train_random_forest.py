# app/train_random_forest.py
# ===============================
# Random Forest classifier for BTC direction
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# CONFIG
# ===============================
DATA_PATH = 'data/processed/features_with_prices_1h.csv'
MODEL_PATH = 'data/processed/rf_1h_classifier.joblib'

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

# ===============================
# FEATURES / TARGET
# ===============================
X = df.drop(columns=['target', 'future_return', 'return'])
y = df['target']

X = X.fillna(0)

# Temporal split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ===============================
# TRAIN MODEL
# ===============================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# EVALUATION
# ===============================
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, pred))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
