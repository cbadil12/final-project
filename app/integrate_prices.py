# app/integrate_prices.py
# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import logging

# 2. Third-party libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

PRICES_PATH = 'data/raw/btcusd-1h.csv'
FNG_PATH = 'data/interim/fear_greed.csv'
AGG_DIR = 'data/processed/'
OUTPUT_DIR = 'data/processed/'

WINDOWS = ['4h', '1h']

# ===============================
# LOAD PRICES
# ===============================
def load_prices():
    df = pd.read_csv(PRICES_PATH, sep=';', header=None)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:00:00', utc=True)
    df = df.drop(['date', 'time'], axis=1)
    df = df.set_index('datetime')
    logging.info(f"Loaded prices with {len(df)} rows")
    return df

# ===============================
# RESAMPLE PRICES
# ===============================
def resample_prices(df, window='4h'):
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
# MERGE SENTIMENT
# ===============================
def merge_with_sentiment(prices, window):
    agg_path = os.path.join(AGG_DIR, f'aggregated_{window}.csv')
    if not os.path.exists(agg_path):
        logging.error(f"Aggregated file not found: {agg_path}")
        return None
    
    agg = pd.read_csv(agg_path)
    agg['published_at'] = pd.to_datetime(agg['published_at'], utc=True)
    agg = agg.set_index('published_at')
    
    merged = prices.join(agg, how='left')
    merged = merged.fillna(0)
    logging.info(f"Merged with sentiment for {window}: {len(merged)} rows")
    return merged

# ===============================
# ADD FNG
# ===============================
def add_fng(merged):
    if not os.path.exists(FNG_PATH):
        logging.warning(f"FNG file not found: {FNG_PATH}")
        return merged
    
    fng = pd.read_csv(FNG_PATH)
    fng['datetime'] = pd.to_datetime(fng['datetime'], utc=True)
    fng = fng.set_index('datetime')
    
    merged = merged.join(fng, how='left')
    merged['fear_greed_index'] = merged['fear_greed_index'].fillna(50)
    logging.info("Added FNG")
    return merged

# ===============================
# TRAIN MODEL
# ===============================
def train_model(merged, window, classification=False):
    features = [c for c in merged.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'return']]
    X = merged[features]
    y = merged['return']
    
    if classification:
        y = (y > 0).astype(int)  # Up/down
        model = XGBClassifier(random_state=42)
        metric_fn = accuracy_score
        metric_name = 'Accuracy'
    else:
        model = XGBRegressor(random_state=42)
        metric_fn = mean_squared_error
        metric_name = 'MSE'
    
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    metric = metric_fn(y_test, preds)
    logging.info(f"XGBoost {metric_name} for {window}: {metric:.6f}")
    
    # Save model (use booster for safety)
    model_path = os.path.join(OUTPUT_DIR, f'xgb_{window}.json')
    model.get_booster().save_model(model_path)
    logging.info(f"Model saved to {model_path}")
    
    return metric

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    prices = load_prices()
    
    for w in WINDOWS:
        resampled = resample_prices(prices, w)
        merged = merge_with_sentiment(resampled, w)
        if merged is None:
            continue
        merged = add_fng(merged)
        
        output_path = os.path.join(OUTPUT_DIR, f'features_with_prices_{w}.csv')
        merged.to_csv(output_path)
        logging.info(f"Saved merged for {w} to {output_path} with {len(merged)} rows")
        
        # Train regression
        train_model(merged, w, classification=False)
        
        # Optional: Train classification (uncomment if you want up/down)
        # train_model(merged, w, classification=True)