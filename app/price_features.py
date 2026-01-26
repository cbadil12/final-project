# app/price_features.py
# =====================================
# Build PRICE-only feature dataset (1h/4h/24h) - TRAINING-COMPAT VERSION
# Input : data/raw/raw_price_dowloaded_*.csv (semicolon separated)
# Output: data/interim/price_features_*.csv
#
# Exposes:
#   get_price_features_row_nearest(resolution, target_ts, raw_path_override=None)
#   -> returns a single-row DF with features nearest to target_ts (UTC)
# =====================================

import os
import logging
import argparse
from functools import lru_cache

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_1H = "data/raw/raw_price_dowloaded_1h_START_10-01-2015_END_18-01-2026.csv"
RAW_4H = "data/raw/raw_price_dowloaded_4h_START_31-12-2011_END_20-12-2025.csv"
RAW_24H = "data/raw/raw_price_dowloaded_24h_START_10-01-2015_END_18-01-2026.csv"

OUT_1H = "data/interim/price_features_1h.csv"
OUT_4H = "data/interim/price_features_4h.csv"
OUT_24H = "data/interim/price_features_24h.csv"

SUPPORTED_RESOLUTIONS = {"1h", "4h", "24h"}

# runtime speed: solo necesitas historial para rolling 24 + lags 4 => ~200 filas sobran
TAIL_ROWS_BY_RES = {"1h": 5000, "4h": 5000, "24h": 3000}


def _paths_for_resolution(resolution: str) -> tuple[str, str]:
    if resolution == "1h":
        return RAW_1H, OUT_1H
    if resolution == "4h":
        return RAW_4H, OUT_4H
    if resolution == "24h":
        return RAW_24H, OUT_24H
    raise ValueError("resolution must be 1h, 4h or 24h")


def _max_delta_for_resolution(resolution: str) -> pd.Timedelta:
    if resolution == "1h":
        return pd.Timedelta("1H")
    if resolution == "4h":
        return pd.Timedelta("4H")
    if resolution == "24h":
        return pd.Timedelta("24H")
    raise ValueError("resolution must be 1h, 4h or 24h")


# ===============================
# LOAD RAW
# ===============================
@lru_cache(maxsize=8)
def load_raw(path: str) -> pd.DataFrame:
    """
    Loads raw price CSV with ';' separator and EU datetime format.
    Expected columns (typical): time;open;max;min;close
    Renames: time->datetime, max->high, min->low
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw price file not found: {path}")

    df = pd.read_csv(path, sep=";")

    # Normalize column names
    df = df.rename(columns={"time": "datetime", "max": "high", "min": "low"})

    if "datetime" not in df.columns:
        raise ValueError(f"Missing datetime column. Got columns: {list(df.columns)}")

    # Parse datetime (EU format) and force UTC
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    needed = {"open", "high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw: {missing}. Got: {list(df.columns)}")

    # Ensure numeric
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing essential prices
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()

    return df


# ===============================
# FEATURE ENGINEERING (TRAINING COMPAT)
# ===============================
def build_features(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """
    EXACT training-compatible features (same column names/pattern as used in training):
    - ret_1, ret_2, ret_4, ret_8, ret_24
    - close_lag_1, close_lag_2, close_lag_4
    - std_8, std_24
    - mom_sma24
    - range_norm
    - hour, dayofweek
    + keeps OHLC (training later drops raw OHLC, but keeping here is safe)
    """
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(f"Unsupported resolution: {resolution}. Use one of {sorted(SUPPORTED_RESOLUTIONS)}")

    out = df[["open", "high", "low", "close"]].copy()

    # Returns (past only)
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_2"] = out["close"].pct_change(2)
    out["ret_4"] = out["close"].pct_change(4)
    out["ret_8"] = out["close"].pct_change(8)
    out["ret_24"] = out["close"].pct_change(24)

    # Lags
    out["close_lag_1"] = out["close"].shift(1)
    out["close_lag_2"] = out["close"].shift(2)
    out["close_lag_4"] = out["close"].shift(4)

    # Rolling volatility (uses past returns)
    out["std_8"] = out["ret_1"].rolling(8, min_periods=8).std()
    out["std_24"] = out["ret_1"].rolling(24, min_periods=24).std()

    # Simple momentum vs SMA24
    sma_24 = out["close"].rolling(24, min_periods=24).mean()
    out["mom_sma24"] = (out["close"] / sma_24) - 1.0

    # Candle range normalized
    out["range_norm"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    # Time features
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek

    return out


# ===============================
# DYNAMIC_PREDICT HELPER
# ===============================
def get_price_features_row_nearest(
    resolution: str,
    target_ts,
    raw_path_override: str | None = None
) -> pd.DataFrame:
    """
    Builds training-compatible features and returns the single row nearest to target_ts (UTC).
    Designed for integration with dynamic_predict.py.
    """
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError("resolution must be 1h, 4h or 24h")

    raw_path, _ = _paths_for_resolution(resolution)
    if raw_path_override:
        raw_path = raw_path_override

    # IMPORTANT: avoid stale cached data when override is used
    if raw_path_override:
        df = load_raw.__wrapped__(raw_path)  # bypass cache
    else:
        df = load_raw(raw_path)

    # speed: keep only recent rows
    tail_n = TAIL_ROWS_BY_RES.get(resolution, 5000)
    df = df.tail(tail_n)

    feats = build_features(df, resolution).dropna().sort_index()

    target_ts = pd.to_datetime(target_ts, utc=True, errors="coerce")
    if pd.isna(target_ts):
        raise ValueError("Invalid target_ts (could not parse to datetime)")

    pos = feats.index.get_indexer([target_ts], method="nearest")[0]
    nearest_ts = feats.index[pos]

    # guard: prevent silently using a very old candle
    max_delta = _max_delta_for_resolution(resolution)
    if abs(nearest_ts - target_ts) > max_delta:
        raise ValueError(
            f"No close timestamp for {target_ts}. Nearest is {nearest_ts} "
            f"(delta={abs(nearest_ts - target_ts)})"
        )

    return feats.iloc[[pos]].copy()


# ===============================
# CLI (batch generation)
# ===============================
def main(resolution: str):
    os.makedirs("data/interim", exist_ok=True)

    raw_path, out_path = _paths_for_resolution(resolution)

    df = load_raw(raw_path)
    feats = build_features(df, resolution)

    feats.to_csv(out_path)
    logging.info(f"Saved: {out_path} | shape={feats.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["1h", "4h", "24h"], default="1h")
    args = parser.parse_args()

    main(args.resolution)
