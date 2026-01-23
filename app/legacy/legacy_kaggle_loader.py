# app/legacy_kaggle_loader.py
# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import sys
import logging
import hashlib

# Add project root to Python path (KEEP SIMPLE / FUNCTIONAL)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Third-party libraries
import pandas as pd

# 3. Local application imports
from src.config.news_keywords import KEYWORD_MAP

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_RAW_DIR = 'data/raw/'
INTERIM_DIR = 'data/interim/'
KAGGLE_FILES = [
    'Bitcoin_Pulse_Hourly_Dataset_from_Markets_Trends_and_Fear.csv',
    'bitcoin_sentiments_21_24.csv',
    'bitcoin_titles.csv',
    'cryptonews.csv',
    'inflation_news_articles.csv',
    'MIT_AI_ARTICLES.csv'
]

MAX_DESC_LEN = 1000

# ===============================
# HELPER FUNCTIONS
# ===============================
def assign_axis(text, file_name):
    text_lower = str(text).lower()

    if 'inflation' in file_name.lower():
        return 'MACRO'
    elif 'ai' in file_name.lower():
        return 'TECH'
    elif 'bitcoin' in file_name.lower() or 'crypto' in file_name.lower():
        return 'BTC'

    # Fallback: try using KEYWORD_MAP terms (best-effort heuristic)
    for axis, keywords in KEYWORD_MAP.items():
        kw_list = [kw.strip("'").lower() for kw in keywords.replace('(', '').replace(')', '').split(' OR ')]
        if any(kw in text_lower for kw in kw_list):
            return axis

    return 'BTC'

def make_hash_id(text):
    return hashlib.md5(str(text).encode('utf-8', errors='ignore')).hexdigest()

# ===============================
# LOAD AND STANDARDIZE EACH FILE
# ===============================
def load_and_standardize(file_path):
    file_name = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {file_name} with {len(df)} rows")

        # Special case: Bitcoin_Pulse (extract Fear & Greed)
        if 'Bitcoin_Pulse' in file_name:
            if 'timestamp' in df.columns and 'fear_greed_index' in df.columns:
                fng_df = df[['timestamp', 'fear_greed_index']].copy()
                fng_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
                fng_df['datetime'] = pd.to_datetime(fng_df['datetime'], utc=True, errors='coerce')
                fng_df.dropna(subset=['datetime'], inplace=True)

                os.makedirs(INTERIM_DIR, exist_ok=True)
                fng_path = os.path.join(INTERIM_DIR, 'fear_greed.csv')
                fng_df.to_csv(fng_path, index=False)
                logging.info(f"Extracted Fear & Greed index to {fng_path} ({len(fng_df)} rows)")
            else:
                logging.warning(f"Missing timestamp/fear_greed_index in {file_name}")

            return pd.DataFrame()

        # Column renaming map
        rename_map = {
            'Date': 'published_at',
            'date': 'published_at',
            'timestamp': 'published_at',
            'published_date': 'published_at',
            'publication_date': 'published_at',  # useful for MIT_AI
            'Short Description': 'description',
            'text': 'description',
            'Title': 'title',
            'title': 'title',
            'Links': 'url',
            'url': 'url',
            'link': 'url',
            'paper_link': 'url',
            'source': 'source',
            'body': 'body',
            'summary': 'summary'
        }

        # MIT_AI: sometimes has duplicate/extra datetime columns
        if 'MIT_AI' in file_name and 'datetime' in df.columns and 'published_at' not in df.columns:
            df['published_at'] = df['datetime']
            df.drop(columns=['datetime'], inplace=True)

        df.rename(columns=rename_map, inplace=True)

        # Drop unnecessary columns
        drop_cols = ['Unnamed: 0', 'id', 'author', 'country', 'language', 'topic']
        df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

        # Ensure published_at exists
        if 'published_at' not in df.columns:
            logging.warning(f"No published_at column in {file_name}")
            return pd.DataFrame()

        # Convert to datetime UTC
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
        df.dropna(subset=['published_at'], inplace=True)
        df = df.reset_index(drop=True)

        # Fill missing core columns
        if 'title' not in df.columns:
            df['title'] = ''
        if 'source' not in df.columns:
            df['source'] = 'Kaggle'
        if 'url' not in df.columns:
            df['url'] = ''

        # Build description
        if 'summary' in df.columns:
            df['description'] = df['summary'].fillna('').astype(str).str[:MAX_DESC_LEN]
        elif 'body' in df.columns:
            df['description'] = df['body'].fillna('').astype(str).str[:MAX_DESC_LEN]
        elif 'description' in df.columns:
            df['description'] = df['description'].fillna('').astype(str).str[:MAX_DESC_LEN]
        else:
            df['description'] = ''

        df.drop(columns=['summary', 'body'], inplace=True, errors='ignore')

        # Create text_nlp
        df['title'] = df['title'].fillna('').astype(str)
        df['description'] = df['description'].fillna('').astype(str)
        df['text_nlp'] = df['title'] + ' ' + df['description']
        df = df[df['text_nlp'].str.strip() != '']
        df = df.reset_index(drop=True)

        # Assign axis
        df['axis'] = df.apply(lambda row: assign_axis(row['text_nlp'], file_name), axis=1)

        # Deduplicate within file (KEEP hash_id)
        df['hash_id'] = df['text_nlp'].apply(make_hash_id)
        df.drop_duplicates(subset='hash_id', inplace=True)
        df = df.reset_index(drop=True)

        # Ensure url exists in return schema
        if 'url' not in df.columns:
            df['url'] = ''

        return df[['published_at', 'title', 'description', 'source', 'axis', 'text_nlp', 'url', 'hash_id']]

    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
        return pd.DataFrame()

# ===============================
# MAIN FUNCTION
# ===============================
def load_all_kaggle():
    os.makedirs(INTERIM_DIR, exist_ok=True)

    dfs = []
    for file in KAGGLE_FILES:
        path = os.path.join(DATA_RAW_DIR, file)
        if os.path.exists(path):
            df = load_and_standardize(path)
            if not df.empty:
                if not df.index.is_unique:
                    logging.warning(f"Non-unique index in {os.path.basename(path)} - Resetting")
                    df = df.reset_index(drop=True)
                dfs.append(df)
        else:
            logging.info(f"{file} not found -> skipping")

    if not dfs:
        logging.warning("No Kaggle data loaded")
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)

    # Global deduplication (use hash_id)
    before_dedup = len(all_df)
    if 'hash_id' not in all_df.columns:
        all_df['hash_id'] = all_df['text_nlp'].apply(make_hash_id)

    all_df.drop_duplicates(subset='hash_id', inplace=True)
    all_df = all_df.reset_index(drop=True)
    logging.info(f"Removed {before_dedup - len(all_df)} global duplicates")

    output_path = os.path.join(INTERIM_DIR, 'kaggle_standardized.csv')
    all_df.to_csv(output_path, index=False)
    logging.info(f"Kaggle standardized data saved to {output_path} with {len(all_df)} articles")

    return all_df

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    load_all_kaggle()
