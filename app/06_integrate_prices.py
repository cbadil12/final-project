# app/06_integrate_prices.py
# ===============================
# IMPORTS
# ===============================

import os
import logging

import pandas as pd
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PRICES_PATH = 'data/raw/btcusd-1h.csv'
FNG_PATH = 'data/interim/fear_greed.csv'
AGG_DIR = 'data/processed/'
OUTPUT_DIR = 'data/processed/'

WINDOWS = ['1h', '4h']

# ===============================
# LOAD AND PREPARE PRICES
# ===============================
def load_prices():
    df = pd.read_csv(PRICES_PATH, sep=';', header=None)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d/%m/%Y %H:00:00', utc=True)
    df = df.drop(['date', 'time'], axis=1)
    df = df.set_index('datetime').sort_index()
    logging.info(f"Loaded prices: {len(df)} rows")
    return df

# ===============================
# RESAMPLE PRICES TO TARGET WINDOW
# ===============================
def resample_prices(prices, window):
    resampled = prices.resample(window).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna(subset=['close'])
    
    resampled['return'] = resampled['close'].pct_change()
    resampled = resampled.dropna(subset=['return'])
    
    logging.info(f"Resampled prices to {window}: {len(resampled)} rows")
    return resampled

# ===============================
# MERGE WITH SENTIMENT AGGREGATES
# ===============================
def merge_sentiment(prices_resampled, window):
    agg_path = os.path.join(AGG_DIR, f'aggregated_{window}.csv')
    if not os.path.exists(agg_path):
        logging.error(f"Aggregated file not found: {agg_path}")
        return None
    
    agg = pd.read_csv(agg_path, index_col=0, parse_dates=True)
    agg.index = pd.to_datetime(agg.index, utc=True)
    
    # Left join: keep all price candles, fill missing sentiment with neutral/NaN
    merged = prices_resampled.join(agg, how='left')
    
    # Fill means with 0 (neutral), counts/std/momentum with NaN or 0 as appropriate
    mean_cols = [c for c in merged.columns if 'mean' in c or 'total_mean' in c]
    merged[mean_cols] = merged[mean_cols].fillna(0)
    
    # Other derived columns (shocks, ratios, etc.) forward-fill if reasonable
    ff_cols = [c for c in merged.columns if any(s in c for s in ['shock', 'momentum', 'div_', 'ratio'])]
    merged[ff_cols] = merged[ff_cols].ffill().fillna(0)
    
    logging.info(f"Merged sentiment for {window}: {len(merged)} rows")
    return merged

# ===============================
# ADD FEAR & GREED INDEX
# ===============================
def add_fear_greed(merged):
    if not os.path.exists(FNG_PATH):
        logging.warning(f"FNG file not found: {FNG_PATH} - skipping")
        return merged
    
    fng = pd.read_csv(FNG_PATH)
    fng['datetime'] = pd.to_datetime(fng['datetime'], utc=True)
    fng = fng.set_index('datetime').sort_index()
    
    # Resample F&G to match price index (forward fill is reasonable here)
    fng_resampled = fng.reindex(merged.index, method='ffill')
    merged['fear_greed_index'] = fng_resampled['fear_greed_index'].fillna(50)  # neutral fallback
    
    logging.info("Fear & Greed added")
    return merged

# ===============================
# MAIN INTEGRATION FLOW
# ===============================
def integrate(window):
    prices = load_prices()
    resampled = resample_prices(prices, window)
    merged = merge_sentiment(resampled, window)
    
    if merged is None:
        return
    
    merged = add_fear_greed(merged)
    
    # Final save
    output_path = os.path.join(OUTPUT_DIR, f'features_with_prices_{window}.csv')
    merged.to_csv(output_path)
    logging.info(f"Final features saved: {output_path} ({len(merged)} rows)")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    for w in WINDOWS:
        integrate(w)
    logging.info("Integration completed for all windows.")