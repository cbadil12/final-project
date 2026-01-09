# app/kaggle_loader.py
# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import sys
import logging
import hashlib

# Add project root to Python path
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
    text_lower = text.lower()
    if 'inflation' in file_name.lower():
        return 'MACRO'
    elif 'ai' in file_name.lower():
        return 'TECH'
    elif 'bitcoin' in file_name.lower() or 'crypto' in file_name.lower():
        return 'BTC'
    
    for axis, keywords in KEYWORD_MAP.items():
        kw_list = [kw.strip("'").lower() for kw in keywords.replace('(', '').replace(')', '').split(' OR ')]
        if any(kw in text_lower for kw in kw_list):
            return axis
    return 'BTC'

# ===============================
# LOAD AND STANDARDIZE EACH FILE
# ===============================
def load_and_standardize(file_path):
    file_name = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {file_name} with {len(df)} rows")
        
        # Special case: Bitcoin_Pulse
        if 'Bitcoin_Pulse' in file_name:
            if 'timestamp' in df.columns and 'fear_greed_index' in df.columns:
                fng_df = df[['timestamp', 'fear_greed_index']].copy()
                fng_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
                fng_df['datetime'] = pd.to_datetime(fng_df['datetime'], utc=True)
                fng_path = os.path.join(INTERIM_DIR, 'fear_greed.csv')
                fng_df.to_csv(fng_path, index=False)
                logging.info(f"Extracted Fear & Greed index to {fng_path}")
            else:
                logging.warning(f"Missing fear_greed_index in {file_name}")
            return pd.DataFrame()
        
        # Column renaming map
        rename_map = {
            'Date': 'published_at',
            'date': 'published_at',
            'timestamp': 'published_at',
            'published_date': 'published_at',
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
        
        # Selective rename for MIT_AI to avoid duplicate 'published_at'
        if 'MIT_AI' in file_name:
            rename_map['publication_date'] = 'published_at'
            if 'datetime' in df.columns:
                df['published_at'] = df['datetime']
                df.drop(columns=['datetime'], inplace=True)  # Drop the extra
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
        df = df.reset_index(drop=True)  # Reset after dropna
        
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
        df['text_nlp'] = df['title'].astype(str) + ' ' + df['description'].astype(str)
        df = df[df['text_nlp'].str.strip() != '']
        df = df.reset_index(drop=True)  # Reset after filter
        
        # Assign axis
        df['axis'] = df.apply(lambda row: assign_axis(row['text_nlp'], file_name), axis=1)
        
        # Deduplicate within file
        df['hash'] = df['text_nlp'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        df.drop_duplicates(subset='hash', inplace=True)
        df = df.reset_index(drop=True)  # Reset after drop_duplicates
        df.drop(columns=['hash'], inplace=True)
        
        return df[['published_at', 'title', 'description', 'source', 'axis', 'text_nlp', 'url']]
        
    except Exception as e:
        logging.error(f"Error processing {file_name}: {e}")
        return pd.DataFrame()

# ===============================
# MAIN FUNCTION
# ===============================
def load_all_kaggle():
    dfs = []
    for file in KAGGLE_FILES:
        path = os.path.join(DATA_RAW_DIR, file)
        if os.path.exists(path):
            df = load_and_standardize(path)
            if not df.empty:
                # Extra check and reset before appending
                if not df.index.is_unique:
                    logging.warning(f"Non-unique index in {os.path.basename(path)} - Resetting")
                    df = df.reset_index(drop=True)
                dfs.append(df)
    
    if not dfs:
        logging.warning("No Kaggle data loaded")
        return pd.DataFrame()
    
    all_df = pd.concat(dfs, ignore_index=True)
    
    # Global deduplication
    all_df['hash'] = all_df['text_nlp'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    before_dedup = len(all_df)
    all_df.drop_duplicates(subset='hash', inplace=True)
    all_df.drop(columns=['hash'], inplace=True)
    logging.info(f"Removed {before_dedup - len(all_df)} global duplicates")
    
    output_path = os.path.join(INTERIM_DIR, 'kaggle_standardized.csv')
    os.makedirs(INTERIM_DIR, exist_ok=True)
    all_df.to_csv(output_path, index=False)
    logging.info(f"Kaggle standardized data saved to {output_path} with {len(all_df)} articles")
    
    return all_df

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    load_all_kaggle()