# app/api_prices_cmc_client.py

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
import requests
import pandas as pd

# ===============================
# CONFIGURATION
# ===============================
BASE_URL = "https://pro-api.coinmarketcap.com"   # CMC Pro API base domain
API_KEY = os.getenv("COINMARKETCAP_API_KEY")
if API_KEY is None:
    raise EnvironmentError("ERROR: COINMARKETCAP_API_KEY not found in environment variables")

HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY
}

# You can identify BTC by:
# - symbol="BTC" (recommended)
# - slug="bitcoin"
# - id=1 (CoinMarketCap internal ID for Bitcoin is commonly 1, but symbol is safer)
SYMBOL = "BTC"
CONVERT = "USD"  # default is USD; you can set other fiat/crypto symbols too

START_DATE = "2025-12-19"
END_DATE = "2026-01-15"

# INTERVAL meaning (we map this to CMC OHLCV "time_period" and "interval")
# - "hourly" -> time_period="hourly", interval="1h"
# - "4h"     -> time_period="hourly", interval="4h"
# - "daily"  -> time_period="daily",  interval="1d"
INTERVAL = "daily"  # "hourly", "4h", or "daily"
PRICE_FIELD = "close"  # "close", "open", "max", or "min"


# ===============================
# UTILITIES
# ===============================
def _to_iso8601_utc(date_str: str) -> str:
    """
    Convert YYYY-MM-DD to an ISO-8601 UTC timestamp string.
    CMC accepts Unix or ISO 8601 for time_start/time_end.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _ensure_not_future(end_date: str) -> None:
    """Defensive check: do not request future ranges."""
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    if end_ts > now_ts:
        raise ValueError("ERROR: END_DATE is in the future. Historical endpoints only support past data.")


def _describe_http_error(response: requests.Response | None) -> tuple[str, bool]:
    """
    Returns (message, should_fallback).
    """
    if response is None:
        return ("ERROR: Request failed before a response was received.", False)

    status_code = response.status_code
    error_code = None
    error_message = ""
    try:
        payload = response.json()
        status = payload.get("status", {})
        error_code = status.get("error_code")
        error_message = status.get("error_message") or ""
    except ValueError:
        pass

    if error_code == 1002:
        return ("ERROR: API key missing or not sent in headers.", False)
    if status_code == 401:
        return ("ERROR: Unauthorized. Check your API key.", False)
    if status_code == 403:
        return ("WARN: Access forbidden. This endpoint may not be in your plan.", True)

    message = f"ERROR: HTTP {status_code}"
    if error_message:
        message += f" - {error_message}"
    return (message, False)


# ===============================
# 1) HISTORICAL OHLCV (may be blocked on Free plan)
# ===============================
def fetch_bitcoin_ohlcv_historical(
    start_date: str,
    end_date: str,
    interval: str = "hourly",
    price_field: str = "close",
    symbol: str = "BTC",
    convert: str = "USD",
    skip_invalid: bool = True,
    count: int | None = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from CoinMarketCap.

    Endpoint:
      GET /v2/cryptocurrency/ohlcv/historical

    Common query parameters (CMC docs / Postman):
      - id: One or more comma-separated CoinMarketCap crypto IDs (e.g. "1,1027")
      - slug: One or more comma-separated slugs (e.g. "bitcoin,ethereum")
      - symbol: One or more comma-separated symbols (e.g. "BTC,ETH")  <-- we use this
      - time_period: "daily" or "hourly" (default "daily")
      - time_start: Unix or ISO 8601 start (EXCLUSIVE for the first bucket)
      - time_end: Unix or ISO 8601 end (INCLUSIVE)
      - count: max number of periods to return (default 10, up to 10000)
      - interval: sampling frequency for time_period (e.g. "1h", "2h", "4h", "1d", "7d", "monthly", "yearly")
      - convert: optional currency conversion (default USD; some plans limit how many converts)
      - convert_id: same as convert but uses CMC IDs (cannot be used with convert)
      - skip_invalid: "true" to skip invalid symbols/ids instead of failing the request

    Notes:
      - Many Free/Basics plans DO NOT include this endpoint. If you get 403 Forbidden,
        it's a plan limitation (expected).
    """

    _ensure_not_future(end_date)

    endpoint = f"{BASE_URL}/v2/cryptocurrency/ohlcv/historical"

    # Map your INTERVAL to CoinMarketCap parameters
    if interval == "hourly":
        time_period = "hourly"
        sample_interval = "1h"
    elif interval == "4h":
        time_period = "hourly"
        sample_interval = "4h"
    elif interval == "daily":
        time_period = "daily"
        sample_interval = "1d"
    else:
        raise ValueError("ERROR: interval must be 'hourly', '4h', or 'daily'")

    # CMC accepts Unix timestamps OR ISO 8601 strings.
    # We'll send ISO strings for readability.
    params = {
        "symbol": symbol,                     # alternatively: "id" or "slug"
        "time_period": time_period,           # "hourly" or "daily"
        "time_start": _to_iso8601_utc(start_date),
        "time_end": _to_iso8601_utc(end_date),
        "interval": sample_interval,          # sampling frequency
        "convert": convert,                   # currency conversion (e.g. USD, EUR, BTC...)
        "skip_invalid": str(skip_invalid).lower(),
    }

    # Optional: limit number of periods (if you want)
    # (If you supply time_start/time_end, you often do NOT need count.)
    if count is not None:
        params["count"] = int(count)

    response = requests.get(endpoint, headers=HEADERS, params=params)

    # If your plan blocks the endpoint, this will likely be 403 Forbidden.
    response.raise_for_status()

    data = response.json()

    # Typical structure:
    # data["data"]["quotes"] -> list of OHLCV entries
    quotes = data["data"]["quotes"]

    rows = []
    for q in quotes:
        # q["time_open"], q["time_close"] are ISO strings
        # q["quote"][convert] has "open/high/low/close/volume/market_cap"
        o = q["quote"][convert]

        rows.append({
            "time_open": q.get("time_open"),
            "time_close": q.get("time_close"),
            "open": o.get("open"),
            "high": o.get("high"),
            "low": o.get("low"),
            "close": o.get("close"),
            "volume": o.get("volume"),
            "market_cap": o.get("market_cap"),
        })

    df = pd.DataFrame(rows)
    df["time_open"] = pd.to_datetime(df["time_open"], utc=True)
    df["time_close"] = pd.to_datetime(df["time_close"], utc=True)
    df = df.sort_values("time_open")

    price_field_map = {
        "close": "close",
        "open": "open",
        "max": "high",
        "min": "low",
    }
    if price_field not in price_field_map:
        raise ValueError("ERROR: price_field must be 'close', 'open', 'max', or 'min'")

    src_price = price_field_map[price_field]
    timestamp_col = "time_open" if price_field == "open" else "time_close"
    df_out = df[[timestamp_col, src_price]].rename(
        columns={timestamp_col: "timestamp", src_price: "price"}
    )
    return df_out


# ===============================
# 2) LATEST PRICE (usually available on Free plan)
# ===============================
def fetch_bitcoin_price_latest(
    symbol: str = "BTC",
    convert: str = "USD",
    aux: str | None = None,
    skip_invalid: bool = True
) -> pd.DataFrame:
    """
    Fetch latest quote (NOT historical) from CoinMarketCap.

    Endpoint:
      GET /v2/cryptocurrency/quotes/latest

    Common query parameters:
      - id: one or more comma-separated CMC IDs
      - slug: one or more comma-separated slugs
      - symbol: one or more comma-separated symbols  <-- we use this
      - convert: comma-separated list of fiat/crypto symbols (extra convert costs credits)
      - convert_id: use CMC IDs instead of symbols (cannot be used with convert)
      - aux: request extra fields:
          num_market_pairs, cmc_rank, date_added, tags, platform,
          max_supply, circulating_supply, total_supply, market_cap_by_total_supply,
          volume_24h_reported, volume_7d, volume_7d_reported, volume_30d, volume_30d_reported,
          is_active, is_fiat
      - skip_invalid: "true" to skip invalid symbols/ids instead of failing
    """
    endpoint = f"{BASE_URL}/v2/cryptocurrency/quotes/latest"

    params = {
        "symbol": symbol,
        "convert": convert,
        "skip_invalid": str(skip_invalid).lower()
    }
    if aux is not None:
        params["aux"] = aux

    response = requests.get(endpoint, headers=HEADERS, params=params)
    response.raise_for_status()

    data = response.json()

    # Structure: data["data"][symbol][0]["quote"][convert]["price"]
    item = data["data"][symbol][0]
    quote = item["quote"][convert]

    df = pd.DataFrame([{
        "timestamp": pd.to_datetime(quote["last_updated"], utc=True),
        "price": quote["price"],
        "volume_24h": quote.get("volume_24h"),
        "market_cap": quote.get("market_cap"),
    }])

    return df


# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 1) Try historical OHLCV (may fail on Free plan)
    try:
        df_price = fetch_bitcoin_ohlcv_historical(
            start_date=START_DATE,
            end_date=END_DATE,
            interval=INTERVAL,
            price_field=PRICE_FIELD,
            symbol=SYMBOL,
            convert=CONVERT,
            skip_invalid=True,
            count=None  # you can set e.g. 2000 if you want to cap results
        )
        output_path = os.path.join(
            raw_dir,
            f"raw_prices_cmc_{INTERVAL}_PRICE_{PRICE_FIELD}_START_{START_DATE}_END_{END_DATE}.csv",
        )
        df_price.to_csv(output_path, index=False)
        print(f"OK: CMC OHLCV historical saved to: {output_path}")
        print(df_price.head())

    except requests.exceptions.HTTPError as e:
        message, should_fallback = _describe_http_error(e.response)
        print(message)
        if not should_fallback:
            raise

        # 2) Fallback: latest quote (usually allowed on Free)
        df_latest = fetch_bitcoin_price_latest(
            symbol=SYMBOL,
            convert=CONVERT,
            aux="cmc_rank,circulating_supply,total_supply,max_supply",
            skip_invalid=True
        )
        output_path = os.path.join(
            raw_dir,
            f"coinmarketcap_prices_{INTERVAL}_START_{START_DATE}_END_{END_DATE}.csv",
        )
        df_latest.to_csv(output_path, index=False)
        print(f"OK: Fallback latest quote saved to: {output_path}")
        print(df_latest)
