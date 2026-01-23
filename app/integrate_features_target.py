# app/integrate_features_target.py

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AGG_PATH = "data/interim/aggregated_1h.csv"
PRICES_PATH = "data/raw/btcusd-1h.csv"

OUTPUT_PATH = "data/processed/dataset_sentiment_target_1h.csv"

# ===============================
# HELPERS
# ===============================
def _ensure_utc_hour_index(df: pd.DataFrame, dt_col_candidates=("datetime", "published_at", "time", "timestamp")) -> pd.DataFrame:
    """
    Returns df with UTC datetime index floored to hour.
    Supports:
      - datetime as index
      - datetime in column ("Unnamed: 0", "published_at", etc.)
    """
    df = df.copy()

    # case 1: already datetime index
    if df.index.name in dt_col_candidates or isinstance(df.index, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(df.index, utc=True, errors="coerce").floor("H")
            df = df[~idx.isna()].copy()
            df.index = idx[~idx.isna()]
            df.index.name = "datetime"
            return df.sort_index()
        except Exception:
            pass

    # case 2: Unnamed: 0 column
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "datetime"})

    # case 3: known column
    dt_col = None
    for c in dt_col_candidates:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None and "datetime" in df.columns:
        dt_col = "datetime"

    if dt_col is None:
        raise ValueError(f"No datetime column found in {df.columns.tolist()}")

    df["datetime"] = pd.to_datetime(df[dt_col], utc=True, errors="coerce").dt.floor("H")
    df = df.dropna(subset=["datetime"]).copy()
    df = df.drop(columns=[dt_col], errors="ignore")
    df = df.set_index("datetime").sort_index()
    return df


def load_prices_close(prices_path: str = "data/raw/btcusd-1h.csv") -> pd.Series:
    """
    Supports:
      A) New format (btcusd-1h.csv consolidated): columns with 'datetime' and 'close'
      B) Old format: sep=';' header=None with date/time
    Returns close Series indexed by UTC hour.
    """
    # Attempt A: new format
    try:
        df = pd.read_csv(prices_path)
        if "datetime" in df.columns and "close" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.floor("H")
            df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
            close = pd.to_numeric(df["close"], errors="coerce")
            close = close.dropna()
            logging.info(f"Loaded prices (new format): {len(close)} rows | {close.index.min()} -> {close.index.max()}")
            return close
    except Exception:
        pass

    # Attempt B: old format
    df = pd.read_csv(prices_path, sep=";", header=None, names=["date", "time", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True, utc=True, errors="coerce").dt.floor("H")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    logging.info(f"Loaded prices (old format): {len(close)} rows | {close.index.min()} -> {close.index.max()}")
    return close

# ===============================
# MAIN INTEGRATION FOR TARGET
# ===============================
def integrate_target(
    agg_df: pd.DataFrame,
    prices_path: str = "data/raw/btcusd-1h.csv",
    horizon: int = 1,
    resolution: str = "1h"
) -> pd.DataFrame:
    """
    Integrates prices to calculate target_up and future_return.
    Input: aggregated DF (with features, including F&G if any)
    Returns DF with added target columns (no F&G here, drop close after calc).
    """
    if agg_df.empty:
        logging.warning("Empty aggregated DF")
        return pd.DataFrame()
    
    agg_df = _ensure_utc_hour_index(agg_df)
    logging.info(f"Aggregated input: {agg_df.shape} | from {agg_df.index.min()} to {agg_df.index.max()}")
    
    close = load_prices_close(prices_path)

    # Merge close ONLY for target
    merged = agg_df.copy()
    merged["close"] = close.reindex(merged.index).ffill()

    # If missing close
    missing_close = merged["close"].isna().sum()
    if missing_close:
        logging.warning(f"Missing close after ffill: {missing_close} rows (will be dropped)")

    # Calculate target
    # For resolution '4h', horizon=1 means shift -1 (which is -4 hours since index is 4h spaced)
    # No adjustment needed if index freq is correct
    merged["future_return"] = merged["close"].shift(-horizon) / merged["close"] - 1.0
    merged["target_up"] = (merged["future_return"] > 0).astype(int)

    # Drop rows without target or close
    merged = merged.dropna(subset=["close", "future_return"]).copy()

    # Drop close (avoid leakage)
    merged = merged.drop(columns=["close"])

    # No imputation here (done in aggregate)

    logging.info(f"Target dataset: {merged.shape} | up rate: {merged['target_up'].mean():.4f}")

    return merged

# ===============================
# ENTRY POINT (for standalone test)
# ===============================
if __name__ == "__main__":
    # Test with sample agg
    agg_test_path = "data/interim/aggregated_1h.csv"
    if os.path.exists(agg_test_path):
        agg_df = pd.read_csv(agg_test_path)
        result = integrate_target(agg_df, resolution="1h")
        # Optional save for test
        # output_path = "data/processed/dataset_sentiment_target_1h.csv"
        # result.reset_index().rename(columns={"index": "datetime"}).to_csv(output_path, index=False)
        # logging.info(f"Saved to {output_path}")
    logging.info("Standalone target integration completed.")