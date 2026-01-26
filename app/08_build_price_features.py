# app/08_build_price_features.py
import os
import logging
import argparse

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_1H = "data/raw/raw_price_dowloaded_1h_START_10-01-2015_END_18-01-2026.csv"
RAW_4H = "data/raw/raw_price_dowloaded_4h_START_31-12-2011_END_20-12-2025.csv"
RAW_24H = "data/raw/raw_price_dowloaded_24h_START_10-01-2015_END_18-01-2026.csv"

OUT_1H = "data/interim/price_features_1h.csv"
OUT_4H = "data/interim/price_features_4h.csv"
OUT_24H = "data/interim/price_features_24h.csv"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # normalize columns
    df = df.rename(
        columns={
            "time": "datetime",
            "max": "high",
            "min": "low",
        }
    )
    # parse datetime (EU format)
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")

    # ensure numeric
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close", "high", "low"])
    return df


def build_features(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    out = df[["open", "high", "low", "close"]].copy()

    # ventanas por resolución
    if resolution == "1h":
        r_windows = [1, 2, 4, 8, 24]
        lag_windows = [1, 2, 4]
        vol_windows = [8, 24]
        sma_w = 24
        add_hour = True
    elif resolution == "4h":
        r_windows = [1, 2, 4, 8, 24]  # 24*4h=96h (4 días) -> ok como baseline
        lag_windows = [1, 2, 4]
        vol_windows = [8, 24]
        sma_w = 24
        add_hour = True
    elif resolution == "24h":
        r_windows = [1, 2, 7, 14, 30]
        lag_windows = [1, 2, 3]
        vol_windows = [7, 14]
        sma_w = 30
        add_hour = False
    else:
        raise ValueError("resolution must be 1h, 4h, or 24h")

    # returns (past-only)
    for w in r_windows:
        out[f"ret_{w}"] = out["close"].pct_change(w)

    # lags
    for w in lag_windows:
        out[f"close_lag_{w}"] = out["close"].shift(w)

    # rolling volatility
    out[f"std_{vol_windows[0]}"] = out["ret_1"].rolling(vol_windows[0], min_periods=vol_windows[0]).std()
    out[f"std_{vol_windows[1]}"] = out["ret_1"].rolling(vol_windows[1], min_periods=vol_windows[1]).std()

    # momentum SMA
    sma = out["close"].rolling(sma_w, min_periods=sma_w).mean()
    out[f"mom_sma{sma_w}"] = (out["close"] / sma) - 1.0

    # range normalized
    out["range_norm"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    # time features
    if add_hour:
        out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek

    return out


def main(resolution: str):
    os.makedirs("data/interim", exist_ok=True)

    if resolution == "1h":
        raw_path, out_path = RAW_1H, OUT_1H
    elif resolution == "4h":
        raw_path, out_path = RAW_4H, OUT_4H
    else:
        raw_path, out_path = RAW_24H, OUT_24H

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = load_raw(raw_path)
    feats = build_features(df, resolution)

    feats.to_csv(out_path)
    logging.info(f"Saved: {out_path} | shape={feats.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["1h", "4h", "24h"], default="1h")
    args = parser.parse_args()
    main(args.resolution)
