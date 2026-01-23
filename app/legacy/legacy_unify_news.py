# app/legacy_unify_news.py
# ===============================
# Unify NewsAPI + Kaggle standardized into a single clean CSV for NLP/Sentiment
# Output: data/processed/unified_noticias_nlp_ready.csv
# ===============================

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

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NEWSAPI_PATH = 'data/raw/news_raw.csv'
KAGGLE_PATH = 'data/interim/kaggle_standardized.csv'

OUTPUT_DIR = 'data/processed/'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'unified_noticias_nlp_ready.csv')

MAX_DESC_LEN = 1000

# ===============================
# HELPERS
# ===============================
def make_hash_id(text):
    return hashlib.md5(str(text).encode('utf-8', errors='ignore')).hexdigest()

def safe_read_csv(path):
    if not os.path.exists(path):
        logging.warning(f"File not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

def standardize_schema(df, default_source='Unknown'):
    """
    Expected output schema:
    published_at (UTC datetime), title, description, source, axis, text_nlp, url, hash_id
    """
    if df.empty:
        return df

    # ---- Date column ----
    if 'published_at' not in df.columns:
        # fallback common variants
        if 'publishedAt' in df.columns:
            df.rename(columns={'publishedAt': 'published_at'}, inplace=True)
        elif 'date' in df.columns:
            df.rename(columns={'date': 'published_at'}, inplace=True)
        elif 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'published_at'}, inplace=True)

    if 'published_at' not in df.columns:
        logging.warning("No published_at column found -> returning empty df")
        return pd.DataFrame()

    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True)
    df.dropna(subset=['published_at'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---- Core text ----
    if 'title' not in df.columns:
        df['title'] = ''
    if 'description' not in df.columns:
        df['description'] = ''

    df['title'] = df['title'].fillna('').astype(str)
    df['description'] = df['description'].fillna('').astype(str).str[:MAX_DESC_LEN]

    # ---- source / axis / url ----
    if 'source' not in df.columns:
        df['source'] = default_source
    else:
        df['source'] = df['source'].fillna(default_source).astype(str)

    if 'axis' not in df.columns:
        df['axis'] = 'BTC'  # default

    if 'url' not in df.columns:
        df['url'] = ''
    df['url'] = df['url'].fillna('').astype(str)

    # ---- text_nlp ----
    if 'text_nlp' not in df.columns:
        df['text_nlp'] = (df['title'] + ' ' + df['description']).astype(str)
    else:
        df['text_nlp'] = df['text_nlp'].fillna('').astype(str)

    df = df[df['text_nlp'].str.strip() != ''].copy()
    df.reset_index(drop=True, inplace=True)

    # ---- hash_id ----
    if 'hash_id' not in df.columns:
        df['hash_id'] = df['text_nlp'].apply(make_hash_id)

    # Ensure final columns exist
    keep_cols = ['published_at', 'title', 'description', 'source', 'axis', 'text_nlp', 'url', 'hash_id']
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ''

    return df[keep_cols]

# ===============================
# MAIN
# ===============================
def unify_news():
    # Load
    news_df = safe_read_csv(NEWSAPI_PATH)
    kaggle_df = safe_read_csv(KAGGLE_PATH)

    logging.info(f"Loaded NewsAPI: {len(news_df)} rows")
    logging.info(f"Loaded Kaggle standardized: {len(kaggle_df)} rows")

    # Standardize
    news_df = standardize_schema(news_df, default_source='NewsAPI')
    kaggle_df = standardize_schema(kaggle_df, default_source='Kaggle')

    if news_df.empty and kaggle_df.empty:
        logging.warning("Both sources empty. Nothing to unify.")
        return pd.DataFrame()

    # Concat
    unified = pd.concat([news_df, kaggle_df], ignore_index=True)
    logging.info(f"Unified before dedup: {len(unified)} rows")

    # Global dedup by hash_id
    before = len(unified)
    unified.drop_duplicates(subset='hash_id', inplace=True)
    unified.reset_index(drop=True, inplace=True)
    logging.info(f"Removed {before - len(unified)} duplicates (global)")

    # Sort by time
    unified.sort_values('published_at', inplace=True)
    unified.reset_index(drop=True, inplace=True)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unified.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Saved unified dataset to {OUTPUT_PATH} with {len(unified)} rows")

    return unified

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    unify_news()
