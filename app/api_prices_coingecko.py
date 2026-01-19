# app/api_prices_client.py

# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import requests
from datetime import datetime, timedelta, timezone
# 2. Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()
# 3. Third-party libraries (from requirements.txt)
import pandas as pd

# ===============================
# CONFIGURATION
# ===============================
BASE_URL = "https://pro-api.coingecko.com/api/v3"
API_KEY = os.getenv("COINGECKO_API_KEY")
COIN_ID = "bitcoin"
VS_CURRENCY = "usd"

if API_KEY is None:
    raise EnvironmentError("❌ COINGECKO_API_KEY not found in environment variables")
HEADERS = {
    "accept": "application/json",
    "x-cg-pro-api-key": API_KEY  # <-- clave
}

START_DATE = "2025-12-19"
END_DATE   = "2026-01-15"
INTERVAL = "hourly" # 'hourly' or 'daily'

# ===============================
# Fetch historical Bitcoin prices
# ===============================
def fetch_bitcoin_price(
    start_date: str,
    end_date: str,
    interval: str = "hourly"
) -> pd.DataFrame:

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    # Defensive check: CoinGecko does not allow future ranges
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    if end_ts > now_ts:
        raise ValueError(
            "❌ END_DATE is in the future. CoinGecko only supports historical data."
        )
    
    endpoint = f"{BASE_URL}/coins/{COIN_ID}/market_chart/range"
    params = {
        "vs_currency": VS_CURRENCY,
        "from": start_ts,
        "to": end_ts,
    }
    response = requests.get(endpoint, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns="timestamp_ms")
    if interval == "hourly":
        df = (
            df.set_index("timestamp")
              .resample("1h")
              .last()
              .reset_index()
        )
    return df

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":

    df_price = fetch_bitcoin_price(
        start_date=START_DATE,
        end_date=END_DATE,
        interval="hourly" # 'hourly' or 'daily'
    )
    output_path = "data/raw/coingecko__prices_" + INTERVAL + "_START_"+START_DATE+"_END_"+END_DATE+ ".csv"
    df_price.to_csv(output_path, index=False)
    print(f"📈 Bitcoin price data saved to: {output_path}")
    print(df_price.head())
