# src/backend/dynamic_predict.py

# ===============================
# IMPORTS
# ===============================
# Standard library
import os
import logging
from datetime import datetime

# Third-party libraries
import pandas as pd

# Local application imports (from app/)
from download_last_fng import download_latest_fng
from fetch_news import fetch_news_by_axis
from compute_sentiment import run_sentiment_analysis
from aggregate_features import aggregate_features
from run_model import load_model, run_prediction

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PROCESSED_DIR = '../data/processed/'  # Relative to app/
FNG_PATH = '../data/raw/fear_greed.csv'
WINDOW_HOURS_DEFAULT = 24
DEFAULT_MODEL = 'xgb_clf'

# ===============================
# HELPER FUNCTIONS
# ===============================
def load_dataset(resolution: str) -> pd.DataFrame:
    """
    Loads the historical preprocessed dataset for the resolution.
    """
    filename = f'dataset_sentiment_target_{resolution}.csv'
    path = os.path.join(DATA_PROCESSED_DIR, filename)
    if not os.path.exists(path):
        logging.warning(f"Historical dataset not found: {path} - fallback to live mode")
        return pd.DataFrame()
    
    df = pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
    logging.info(f"Loaded historical dataset: {len(df)} rows")
    return df

def get_features_from_dataset(df: pd.DataFrame, target_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Gets features row from historical dataset (nearest to target_ts).
    Drops target columns to avoid leakage.
    """
    if df.empty:
        return pd.DataFrame()
    
    idx = (df.index - target_ts).abs().idxmin()
    row = df.loc[[idx]].copy()
    
    # Drop target/leakage cols
    drop_cols = ['target_up', 'future_return']  # Adjust based on your dataset
    row = row.drop(columns=[c for c in drop_cols if c in row.columns], errors='ignore')
    
    logging.info(f"Historical features selected for {target_ts}")
    return row

def get_features_live(
    target_ts: pd.Timestamp,
    resolution: str,
    window_hours: int = WINDOW_HOURS_DEFAULT
) -> pd.DataFrame:
    """
    Builds features in live mode: fetch F&G, news, sentiment, aggregate.
    """
    # Step 1: Update F&G
    download_latest_fng(output_path=FNG_PATH)  # Updates CSV
    
    # Step 2: Fetch news
    df_news = fetch_news_by_axis(use_now=True, window_hours=window_hours)
    
    # Step 3: Compute sentiment
    df_sentiment = run_sentiment_analysis(df_news)
    
    # Step 4: Aggregate features
    freq = '1H' if resolution == '1h' else '4H'
    df_agg = aggregate_features(df_sentiment, freq=freq, include_fng=True, fng_path=FNG_PATH)
    
    # Select nearest row
    if df_agg.empty:
        return pd.DataFrame()
    
    idx = (df_agg.index - target_ts).abs().idxmin()
    row = df_agg.loc[[idx]]
    
    logging.info(f"Live features built for {target_ts}")
    return row

# ===============================
# MAIN DYNAMIC PREDICT FUNCTION
# ===============================
def run_dynamic_predict(
    target_ts: str | pd.Timestamp | datetime,
    resolution: str,
    mode: str = 'auto',
    window_hours: int = WINDOW_HOURS_DEFAULT,
    model_name: str = DEFAULT_MODEL
) -> dict:
    """
    Main function for dynamic prediction.
    Follows the scheme: parse, mode determination, features (hist/live), predict, output dict.
    """
    # Parse and validation
    try:
        target_ts = pd.to_datetime(target_ts, utc=True)
    except Exception as e:
        return {'msg': f"Error parsing timestamp: {e}", 'prediction': None}
    
    if resolution not in ['1h', '4h']:
        return {'msg': "Invalid resolution (must be '1h' or '4h')", 'prediction': None}
    
    # Load historical dataset for mode auto/historical
    df_hist = load_dataset(resolution)
    
    # Determine mode
    mode_used = mode
    if mode == 'auto':
        if not df_hist.empty and df_hist.index.min() <= target_ts <= df_hist.index.max():
            mode_used = 'historical'
        else:
            mode_used = 'live'
    
    # Get features based on mode
    if mode_used == 'historical':
        features_df = get_features_from_dataset(df_hist, target_ts)
        if features_df.empty:
            logging.warning("No historical features - fallback to live")
            mode_used = 'live'
            features_df = get_features_live(target_ts, resolution, window_hours)
    elif mode_used == 'live':
        features_df = get_features_live(target_ts, resolution, window_hours)
    else:
        return {'msg': "Invalid mode (auto, historical, live)", 'prediction': None}
    
    if features_df.empty:
        return {'msg': "No features available", 'prediction': None}
    
    # Load model and predict
    model = load_model(resolution, model_name)
    pred_dict = run_prediction(features_df, target_ts, model)
    
    # Placeholder for fusion (e.g., with ARIMA from Carlos)
    # fused_proba = 0.3 * pred_dict['proba_up'] + 0.7 * carlos_proba_up  # Uncomment when ready
    # pred_dict['fused_prediction'] = 1 if fused_proba > 0.5 else 0
    # pred_dict['fused_confidence'] = max(fused_proba, 1 - fused_proba)
    
    # Final output dict
    output = {
        'prediction': pred_dict['prediction'],
        'confidence': pred_dict['confidence'],
        'proba_up': pred_dict['proba_up'],
        'mode_used': mode_used,
        'timestamp': str(target_ts),
        'resolution': resolution,
        'model': model_name,
        'msg': 'Success'
    }
    
    logging.info(f"Dynamic predict completed: {output}")
    return output

# ===============================
# ENTRY POINT (for standalone test)
# ===============================
if __name__ == "__main__":
    # Test with current time
    test_ts = datetime.utcnow()
    test_resolution = '1h'
    test_mode = 'auto'
    
    result = run_dynamic_predict(test_ts, test_resolution, test_mode)
    print("\nDynamic Predict Test Result:")
    print(result)