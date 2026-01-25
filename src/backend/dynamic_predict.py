# src/backend/dynamic_predict.py

# ===============================
# IMPORTS
# ===============================
# Standard library
import os
import logging
from datetime import datetime
from pathlib import Path


# Third-party libraries
import pandas as pd

# Local application imports (from app/)
from app.download_last_fng import download_latest_fng
from app.compute_sentiment import run_sentiment_analysis
from app.aggregate_features import aggregate_features
from app.run_model import load_model, run_prediction

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../src/backend/dynamic_predict.py -> /src -> / (root)
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FNG_PATH = PROJECT_ROOT / "data" / "raw" / "fear_greed.csv"
WINDOW_HOURS_DEFAULT = 24
DEFAULT_MODEL = 'xgb_clf'

# ===============================
# HELPER FUNCTIONS
# ===============================
def load_dataset(resolution: str) -> pd.DataFrame:
    """
    Loads the historical preprocessed dataset for the resolution.
    Accepts either 'timestamp' or 'datetime' as time column.
    Returns a DF indexed by timestamp (UTC), sorted.
    """

    # Ruta robusta (independiente del cwd)
    project_root = Path(__file__).resolve().parents[2]
    data_processed_dir = project_root / "data" / "processed"

    filename = f'dataset_sentiment_target_{resolution}.csv'
    path = data_processed_dir / filename

    if not path.exists():
        logging.warning(f"Historical dataset not found: {path} - fallback to live mode")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Detectar columna de tiempo
    if "timestamp" in df.columns:
        time_col = "timestamp"
    elif "datetime" in df.columns:
        time_col = "datetime"
    else:
        logging.error(f"Dataset has no 'timestamp' or 'datetime' column: {path}")
        return pd.DataFrame()

    # Normalizar a timestamp UTC y poner índice
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    logging.info(f"Loaded historical dataset: {len(df)} rows (time_col={time_col})")
    return df


def get_features_from_dataset(df: pd.DataFrame, target_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Gets features row from historical dataset (nearest to target_ts).
    Drops target columns to avoid leakage.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Asegura índice ordenado (requisito para nearest)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Encontrar el índice más cercano sin usar .abs()
    pos = df.index.get_indexer([target_ts], method="nearest")[0]
    if pos == -1:
        return pd.DataFrame()

    row = df.iloc[[pos]].copy()

    # Drop target/leakage cols
    drop_cols = ['target_up', 'future_return']
    row = row.drop(columns=[c for c in drop_cols if c in row.columns], errors='ignore')

    logging.info(f"Historical features selected for {target_ts} -> {row.index[0]}")
    return row

def get_features_live(
    target_ts: pd.Timestamp,
    resolution: str,
    window_hours: int = WINDOW_HOURS_DEFAULT
) -> pd.DataFrame:
    """
    Builds features in live mode: fetch F&G, news, sentiment, aggregate.
    Live requiere dependencias (newsapi) y NEWS_API_KEY; si no existen, devuelve DF vacío.
    """
    # --- Lazy imports - si no esta la NEWS API disponible ---
    try:
        # si tus módulos están en app/
        from app.fetch_news import fetch_news_by_axis
    except Exception:
        # fallback por si tu proyecto resuelve módulos sin prefijo app.
        try:
            from fetch_news import fetch_news_by_axis
        except ModuleNotFoundError as e:
            logging.warning(f"Live mode temporarily unavailable (missing dependency): {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.warning(f"Live mode temporarily unavailable (fetch_news import error): {e}")
            return pd.DataFrame()

    # Step 1: Update F&G
    download_latest_fng(output_path=FNG_PATH)  # Updates CSV
    
    # Step 2: Fetch news
    try:
        df_news = fetch_news_by_axis(use_now=True, window_hours=window_hours)
    except Exception as e:
        logging.warning(f"Live mode temporarily unavailable (fetch_news_by_axis failed): {e}")
        return pd.DataFrame()
    
    # Step 3: Compute sentiment
    df_sentiment = run_sentiment_analysis(df_news)
    
    # Step 4: Aggregate features
    freq = '1H' if resolution == '1h' else '4H'
    df_agg = aggregate_features(df_sentiment, freq=freq, include_fng=True, fng_path=FNG_PATH)
    
    # Select nearest row
    if df_agg.empty:
        return pd.DataFrame()
    
    # Select nearest row (sin .abs)
    if not df_agg.index.is_monotonic_increasing:
        df_agg = df_agg.sort_index()

    pos = df_agg.index.get_indexer([target_ts], method="nearest")[0]
    if pos == -1:
        return pd.DataFrame()

    row = df_agg.iloc[[pos]]
    
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