# app/api_news_client.py

# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import time
from datetime import datetime, timedelta
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
print (NEWS_API_KEY)
if NEWS_API_KEY is None:
    raise EnvironmentError("❌ NEWS_API_KEY not found in environment variables")

START_DATE = "2025-12-05"
END_DATE = "2025-12-12"
LANGUAGE = "en"

# ===============================
# INITIALIZE CLIENT
# ===============================
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


print(newsapi)

# ===============================
# Test connection with NewsAPI using a minimal query.
# ===============================
def test_connection(      
) -> bool:
    
    print("📡 Testing connection to NewsAPI...")
    try:
        response = newsapi.get_everything(
            q="Bitcoin",
            language=LANGUAGE,
            page_size=5
        )
        if response["status"] == "ok":
            print(f"✅ Connection successful. Total results: {response['totalResults']}")
            first_article = response["articles"][0]
            print("\n📰 Sample article retrieved:")
            print(f"- Title: {first_article['title']}")
            print(f"- Published at: {first_article['publishedAt']}")
            return True
        print("❌ API response returned an error.")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

# ===============================
# Fetch news articles day by day for each keyword axis
# ===============================
def fetch_news_by_axis(
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    
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
                print(f"  ❌ ERROR for axis {axis_name} on {day_str}: {e}")
                time.sleep(5)
        current_date += timedelta(days=1)
    print("\n--- NEWS EXTRACTION COMPLETED ---")
    return pd.DataFrame(all_news)

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    if test_connection():
        df_news = fetch_news_by_axis(
            start_date=START_DATE,
            end_date=END_DATE
        )
        df_news["published_at"] = pd.to_datetime(df_news["published_at"], utc=True)
        output_path = "data/raw/news_raw.csv"
        df_news.to_csv(output_path, index=False)
        print(f"\n📁 Raw news data saved to: {output_path}")
        print(f"Total articles collected: {len(df_news)}")
