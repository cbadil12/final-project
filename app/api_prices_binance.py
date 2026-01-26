# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
from datetime import datetime, timedelta, timezone
# 2. Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()
# 3. Third-party libraries (from requirements.txt)
import pandas as pd
from binance.client import Client

# ===============================
# INITIALIZE CLIENT
# ===============================
def _get_prices_client():
    API_KEY = os.getenv("BINANCE_API_KEY")
    API_SECRET = os.getenv("BINANCE_API_SECRET")
    if API_KEY is None:
        raise EnvironmentError("ERROR: BINANCE_API_KEY not found in environment variables")
    if API_SECRET is None:
        raise EnvironmentError("ERROR: BINANCE_API_SECRET not found in environment variables")
    try:
        from binance.client import Client
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Missing dependency 'binance.client'. Install pip package: binance.client"
        )
    return Client(API_KEY, API_SECRET)

# ===============================
# FETCH PRICES
# ===============================
def fetch_prices_by_axis(
    start_date: str = None,
    end_date: str = None,
    interval: str = "hourly",
    use_now: bool = False,
)-> pd.DataFrame:
    # Interval map
    INTERVAL_MAP = {
    "hourly": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "daily": Client.KLINE_INTERVAL_1DAY,
    }
    # Check interval
    if interval not in INTERVAL_MAP:
        raise ValueError("ERROR: INTERVAL must be 'hourly', '4h', or 'daily'")
    # Security check for dates
    if use_now:
        now_dt = datetime.now(timezone.utc)
        delta_map = {
            "hourly": timedelta(hours=24),
            "4h": timedelta(hours=48),
            "daily": timedelta(days=30),
        }
        start_dt = now_dt - delta_map[interval]
        start_date = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_date = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date required if not use_now")
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)
    if end_dt <= start_dt:
        raise ValueError("end_date must be later than start_date")
    # Set client
    client = _get_prices_client()
    # Set interval
    binance_interval = INTERVAL_MAP[interval]
    # Get OHLC prices for Bitcoin
    klines = client.get_historical_klines(
        "BTCUSD",
        binance_interval,
        start_date,
        end_date,
    )
    # Set column names
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
    df_price = df_price.rename(columns={"open_time": "time"})
    df_price = df_price[["time", "open", "high", "low", "close"]]
    return df_price

# ===============================
# ENTRY POINT (FOR STANDALONE TESTS)
# ===============================
if __name__ == "__main__":
    START_DATE = "2026-01-05"
    END_DATE = "2026-01-10"
    INTERVAL = "hourly"  # "hourly", "4h", or "daily"
    df_prices=(fetch_prices_by_axis(start_date=START_DATE, end_date=END_DATE, interval=INTERVAL))
    print(df_prices)
    output_path = f"data/raw/binance_prices_raw_{INTERVAL}_START_{START_DATE}_END_{END_DATE}.csv"
    df_prices.to_csv(output_path, index=False, sep=";")