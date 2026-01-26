# app/09_build_price_dataset.py
import os
import logging
import argparse

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IN_1H = "data/interim/price_features_1h.csv"
IN_4H = "data/interim/price_features_4h.csv"
IN_24H = "data/interim/price_features_24h.csv"

OUT_1H = "data/processed/dataset_price_target_1h.csv"
OUT_4H = "data/processed/dataset_price_target_4h.csv"
OUT_24H = "data/processed/dataset_price_target_24h.csv"


def build_dataset(in_path: str, out_path: str):
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(in_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notnull()].sort_index()

    # target using t+1
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target_up"] = (df["future_return"] > 0).astype(int)

    # drop last row (no future) + any NaNs from rolling/lags
    before = len(df)
    df = df.dropna()
    after = len(df)

    df.to_csv(out_path)
    logging.info(f"Saved: {out_path} | shape={df.shape} | dropped={before-after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["1h", "4h", "24h"], default="1h")
    args = parser.parse_args()

    if args.resolution == "1h":
        build_dataset(IN_1H, OUT_1H)
    elif args.resolution == "4h":
        build_dataset(IN_4H, OUT_4H)
    elif args.resolution == "24h":
        build_dataset(IN_24H, OUT_24H)
    else:
        raise ValueError("resolution must be 1h, 4h or 24h")

