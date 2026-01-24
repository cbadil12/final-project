# app/fetch_news.py

# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import sys 
import time
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# 2. Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# 3. Third-party libraries (from requirements.txt)
import pandas as pd
from newsapi import NewsApiClient

# 4. Local application imports
from src.config.news_keywords import KEYWORD_MAP

# ===============================
# CONFIGURATION
# ===============================
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
if NEWS_API_KEY is None:
    raise EnvironmentError("NEWS_API_KEY not found in environment variables")

LANGUAGE = "en"

# ===============================
# INITIALIZE CLIENT
# ===============================
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# ===============================
# TEST CONNECTION
# ===============================
def test_connection():
    print("Testing connection to NewsAPI...")
    try:
        response = newsapi.get_everything(
            q="Bitcoin",
            language=LANGUAGE,
            page_size=5
        )
        if response["status"] == "ok":
            print(f"Connection successful. Total results: {response['totalResults']}")
            first_article = response["articles"][0]
            print("\nSample article retrieved:")
            print(f"- Title: {first_article['title']}")
            print(f"- Published at: {first_article['publishedAt']}")
            return True
        print("API response returned an error.")
        return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

# ===============================
# FETCH NEWS
# ===============================
def fetch_news_by_axis(
    start_date: str = None,
    end_date: str = None,
    use_now: bool = False,
    window_hours: int = 24
) -> pd.DataFrame:
    if use_now:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=window_hours)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")
    
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date required if not use_now")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    all_news = []
    current_date = start_dt
    print("\n--- STARTING NEWS EXTRACTION ---")
    print(f"DATE RANGE: {start_date} to {end_date}")
    while current_date < end_dt:
        day_str = current_date.strftime("%Y-%m-%d")
        print(f"\nProcessing day: {day_str}")
        for axis_name, query in KEYWORD_MAP.items():
            try:
                response = newsapi.get_everything(
                    q=query,
                    language=LANGUAGE,
                    from_param=day_str,
                    to=day_str,
                    sort_by="relevancy",
                    page_size=100,
                )
                if response["status"] == "ok":
                    count = len(response["articles"])
                    print(f"  - {axis_name}: {count} articles found")
                    for article in response["articles"]:
                        all_news.append({
                            "published_at": article["publishedAt"],
                            "title": article["title"],
                            "description": article["description"],
                            "source": article["source"]["name"],
                            "axis": axis_name,
                        })
                time.sleep(0.5)
            except Exception as e:
                print(f"  ERROR for axis {axis_name} on {day_str}: {e}")
                time.sleep(5)
        current_date += timedelta(days=1)
    print("\n--- NEWS EXTRACTION COMPLETED ---")
    return pd.DataFrame(all_news)

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    if test_connection():
        # --- DYNAMIC MODE (Default) ---
        # This mode is used by the app to predict "right now"
        df_news = fetch_news_by_axis(use_now=True, window_hours=24)
        
        # --- STATIC / MANUAL MODE (Optional) ---
        # If you want to download specific dates and save a CSV:
        # 1. Comment out the 'df_news' line above (add a #).
        # 2. Uncomment the lines below (remove the #).
        # 3. Set your manual dates and run the script.
        
        # start_manual = "2026-01-17"
        # end_manual   = "2026-01-21"
        # df_news = fetch_news_by_axis(start_date=start_manual, end_date=end_manual)
        # output_path = f"data/raw/news_raw_{start_manual}_{end_manual}.csv"
        # df_news.to_csv(output_path, index=False)
        # print(f"📁 Historical file saved to: {output_path}")

        # Common processing
        if not df_news.empty:
            df_news["published_at"] = pd.to_datetime(df_news["published_at"], utc=True)
            print(f"Total articles collected: {len(df_news)}")
        else:
            print("No articles were found for the selected range.")