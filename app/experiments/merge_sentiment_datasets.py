# app/merge_sentiment_datasets.py
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

# Files to merge
CSV_FILES = [
    'data/interim/2025.12.17.noticias_raw_sentimiento.csv',
    'data/interim/2026.01.08_noticias_raw_sentimiento.csv',
    'data/raw/12.11.25-12.12.25.noticias_raw_sentimiento.csv',
    'data/interim/kaggle_standardized.csv'
]

OUTPUT_DIR = 'data/processed/'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'merged_sentiment_master.csv')

# ===============================
# MERGE AND STANDARDIZE
# ===============================
def merge_datasets():
    dfs = []
    for file_path in CSV_FILES:
        full_path = os.path.join('/workspaces/final-project', file_path)
        if not os.path.exists(full_path):
            logging.warning(f"File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(full_path)
            logging.info(f"Loaded {file_path} with {len(df)} rows and columns: {list(df.columns)}")
            
            # Find date column: 'publishedAt' or 'published_at'
            date_col = None
            if 'publishedAt' in df.columns:
                date_col = 'publishedAt'
            elif 'published_at' in df.columns:
                date_col = 'published_at'
            
            if date_col is None:
                logging.warning(f"No date column found in {file_path} - Skipping")
                continue
            
            # Rename to standard 'published_at'
            df.rename(columns={date_col: 'published_at'}, inplace=True)
            
            # Convert to UTC datetime
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
            df.dropna(subset=['published_at'], inplace=True)
            logging.info(f"After date cleaning: {len(df)} rows")
            
            # Ensure title and description
            df['title'] = df.get('title', '').fillna('').astype(str)
            df['description'] = df.get('description', '').fillna('').astype(str)
            
            # Create text_nlp if missing
            if 'text_nlp' not in df.columns:
                df['text_nlp'] = df['title'] + ' ' + df['description']
            
            # Ensure source and axis
            if 'source' not in df.columns:
                df['source'] = os.path.basename(file_path)
            if 'axis' not in df.columns:
                df['axis'] = 'BTC'  # default
            
            # Keep only needed columns
            cols = ['published_at', 'title', 'description', 'source', 'axis', 'text_nlp']
            if 'url' in df.columns:
                cols.append('url')
            
            df = df[cols]
            
            dfs.append(df)
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    
    if not dfs:
        logging.warning("No valid data loaded")
        return pd.DataFrame()
    
    # Concatenate
    master_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Combined all sources: {len(master_df)} rows")
    
    # Global deduplication by text content
    master_df['hash'] = master_df['text_nlp'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    before = len(master_df)
    master_df.drop_duplicates(subset='hash', inplace=True)
    master_df.drop(columns=['hash'], inplace=True)
    logging.info(f"Removed {before - len(master_df)} duplicates")
    
    # Sort by date
    master_df.sort_values('published_at', inplace=True)
    master_df.reset_index(drop=True, inplace=True)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Merged master saved to {OUTPUT_PATH} with {len(master_df)} articles")
    
    return master_df

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    merge_datasets()