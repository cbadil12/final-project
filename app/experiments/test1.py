# app/integrate_prices_classifier.py
# ==================================
# Predict BTC price direction (up/down) using prices + sentiment + F&G
# ==================================

# ===============================
# IMPORTS
# ===============================
import os
import logging
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PRICES_PATH = 'data/raw/btcusd-1h.csv'
FNG_PATH = 'data/interim/fear_greed.csv'
AGG_DIR = 'data/processed/'
OUTPUT_DIR = 'data/processed/'

WINDOWS = ['1h', '4h']

# Target config
RETURN_THRESHOLD = 0.001  # 0.1% move considered "real" up

# ===============================
# LOAD PRICES
# ===============================
def load_prices():
    df = pd.read_csv(PRICES_PATH, sep=';', header=None)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%d/%m/%Y %H:00:00',
        utc=True
    )
    df = df.drop(['date', 'time'], axis=1)
    df = df.set_index('datetime')
    logging.info(f"Loaded prices with {len(df)} rows (hourly OHLCV)")
    return df

# ===============================
# RESAMPLE PRICES
# ===============================
def resample_prices(df, window):
    resampled = df.resample(window).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['close'])

    resampled['return'] = resampled['close'].pct_change()
    resampled = resampled.dropna(subset=['return'])

    logging.info(f"Resampled to {window}: {len(resampled)} rows")
    return resampled

# ===============================
# MERGE WITH SENTIMENT
# ===============================
def merge_with_sentiment(prices, window):
    agg_path = os.path.join(AGG_DIR, f'aggregated_{window}.csv')

    if not os.path.exists(agg_path):
        logging.error(f"Aggregated file not found: {agg_path}")
        return None

    agg = pd.read_csv(agg_path)
    agg['published_at'] = pd.to_datetime(agg['published_at'], utc=True)
    agg = agg.set_index('published_at')

    merged = prices.join(agg, how='left').fillna(0)
    logging.info(f"Merged with sentiment for {window}: {len(merged)} rows")
    return merged

# ===============================
# ADD FEAR & GREED
# ===============================
def add_fng(df):
    if not os.path.exists(FNG_PATH):
        logging.warning(f"FNG file not found: {FNG_PATH}")
        return df

    fng = pd.read_csv(FNG_PATH)
    fng['datetime'] = pd.to_datetime(fng['datetime'], utc=True)
    fng = fng.set_index('datetime')

    df = df.join(fng, how='left')
    df['fear_greed_index'] = df['fear_greed_index'].fillna(50)

    logging.info("Added Fear & Greed index")
    return df

# ===============================
# FEATURE ENGINEERING
# ===============================
def add_lag_features(df):
    for lag in [1, 2, 3, 6, 12]:
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    return df

def add_rolling_features(df):
    df['volatility_6'] = df['close'].pct_change().rolling(6).std()
    df['volatility_12'] = df['close'].pct_change().rolling(12).std()
    df['momentum_6'] = df['close'] / df['close'].shift(6) - 1
    df['momentum_12'] = df['close'] / df['close'].shift(12) - 1
    return df

# ===============================
# CREATE TARGET
# ===============================
def create_target(df):
    df['future_return'] = df['close'].pct_change().shift(-1)
    df['target'] = (df['future_return'] > RETURN_THRESHOLD).astype(int)
    df = df.dropna(subset=['target'])
    return df

# ===============================
# TRAIN MODEL
# ===============================
def train_model(df, window):
    # Define features / target
    X = df.drop(columns=['target', 'future_return', 'return'])
    y = df['target']

    X = X.fillna(0)

    # Temporal split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    logging.info(f"Accuracy for {window}: {acc:.4f}")
    logging.info("\n" + classification_report(y_test, preds))

    model_path = os.path.join(OUTPUT_DIR, f'xgb_{window}_classifier.json')
    model.get_booster().save_model(model_path)
    logging.info(f"Model saved to {model_path}")

    return acc

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    prices = load_prices()

    for w in WINDOWS:
        df = resample_prices(prices, w)
        df = merge_with_sentiment(df, w)
        if df is None:
            continue

        df = add_fng(df)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = create_target(df)

        output_path = os.path.join(OUTPUT_DIR, f'features_with_prices_{w}.csv')
        df.to_csv(output_path)
        logging.info(f"Saved dataset for {w} to {output_path} ({len(df)} rows)")

        train_model(df, w)
