# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
from datetime import datetime, timezone
# 2. Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()
# 3. Third-party libraries (from requirements.txt)
import pandas as pd
from binance.client import Client

# ===============================
# CONFIGURATION
# ===============================
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
if API_KEY is None:
    raise EnvironmentError("ERROR: BINANCE_API_KEY not found in environment variables")
if API_SECRET is None:
    raise EnvironmentError("ERROR: BINANCE_API_SECRET not found in environment variables")

START_DATE = "2025-12-20"
END_DATE = "2026-01-18"
INTERVAL = "daily"  # "hourly", "4h", or "daily"

INTERVAL_MAP = {
    "hourly": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "daily": Client.KLINE_INTERVAL_1DAY,
}

def main():
    client = Client(API_KEY, API_SECRET)

    if INTERVAL not in INTERVAL_MAP:
        raise ValueError("ERROR: INTERVAL must be 'hourly', '4h', or 'daily'")
    binance_interval = INTERVAL_MAP[INTERVAL]

    klines = client.get_historical_klines(
        "BTCUSD",
        binance_interval,
        START_DATE,
        END_DATE,
    )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df_price = pd.DataFrame(klines, columns=columns)
    df_price["open_time"] = pd.to_datetime(df_price["open_time"], unit="ms", utc=True)
    df_price["close_time"] = pd.to_datetime(df_price["close_time"], unit="ms", utc=True)
    df_price = df_price.rename(columns={
        "open_time": "time",
        "high": "max",
        "low": "min",
        "volume": "vol",
    })
    df_price = df_price[["time", "open", "max", "min", "close", "vol"]]

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(
        raw_dir,
        f"binance_prices_{INTERVAL}_START_{START_DATE}_END_{END_DATE}.csv",
    )
    df_price.to_csv(output_path, index=False, sep=";")


if __name__ == "__main__":
    main()
