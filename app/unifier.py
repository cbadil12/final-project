# app/unifier.py
# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import logging
import hashlib

# 2. Third-party libraries
import pandas as pd

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NEWS_RAW_PATH = 'data/raw/news_raw.csv'
KAGGLE_STD_PATH = 'data/interim/kaggle_standardized.csv'
PROCESSED_DIR = 'data/processed/'
MASTER_PATH = os.path.join(PROCESSED_DIR, 'news_master.csv')

# ===============================
# UNIFY SOURCES
# ===============================
def unify_sources():
    try:
        # Load NewsAPI data
        if os.path.exists(NEWS_RAW_PATH):
            news_df = pd.read_csv(NEWS_RAW_PATH)
            news_df['published_at'] = pd.to_datetime(news_df['published_at'], utc=True)
            news_df['text_nlp'] = news_df['title'].astype(str) + ' ' + news_df['description'].astype(str)
            logging.info(f"Loaded NewsAPI data with {len(news_df)} articles")
        else:
            news_df = pd.DataFrame()
            logging.warning("No NewsAPI raw file found")
        
        # Load standardized Kaggle data
        if os.path.exists(KAGGLE_STD_PATH):
            kaggle_df = pd.read_csv(KAGGLE_STD_PATH)
            kaggle_df['published_at'] = pd.to_datetime(kaggle_df['published_at'], utc=True)
            logging.info(f"Loaded Kaggle standardized data with {len(kaggle_df)} articles")
        else:
            kaggle_df = pd.DataFrame()
            logging.warning("No Kaggle standardized file found")
        
        # Combine both sources
        master_df = pd.concat([news_df, kaggle_df], ignore_index=True)
        
        if master_df.empty:
            logging.warning("No data to unify")
            return pd.DataFrame()
        
        # Global deduplication using text content
        master_df['hash'] = master_df['text_nlp'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        before_dedup = len(master_df)
        master_df.drop_duplicates(subset='hash', inplace=True)
        master_df.drop(columns=['hash'], inplace=True)
        logging.info(f"Removed {before_dedup - len(master_df)} duplicate articles")
        
        # Sort by publication date
        master_df.sort_values('published_at', inplace=True)
        master_df.reset_index(drop=True, inplace=True)
        
        # Save unified dataset
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        master_df.to_csv(MASTER_PATH, index=False)
        logging.info(f"Unified master dataset saved to {MASTER_PATH} with {len(master_df)} articles")
        
        return master_df
        
    except Exception as e:
        logging.error(f"Error during unification: {e}")
        return pd.DataFrame()

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    unify_sources()