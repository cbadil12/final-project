# app/download_last_fng.py
import pandas as pd
import requests
import os
import logging

def download_latest_fng(output_path="data/raw/fear_greed.csv"):
    """
    Downloads the latest 100 days of Fear & Greed index and updates the local CSV.
    """
    url = "https://api.alternative.me/fng/?limit=100"
    try:
        response = requests.get(url)
        data = response.json()['data']
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to readable date
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df[['datetime', 'value']].rename(columns={'value': 'fear_greed_index'})
        
        # Save to your project folder
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"✅ F&G updated successfully in {output_path}")
        return df
    except Exception as e:
        logging.error(f"❌ Error updating F&G: {e}")
        return None

if __name__ == "__main__":
    download_latest_fng()