# app/sentiment.py
# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import logging

# 2. Third-party libraries
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from joblib import Parallel, delayed
import numpy as np
import torch

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = 'data/processed/merged_sentiment_master.csv'
OUTPUT_DIR = 'data/processed/'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'news_with_sentiment.csv')

# Model configuration
FINBERT_MODEL = 'ProsusAI/finbert'
BATCH_SIZE = 32  # Adjust based on your memory/GPU

# ===============================
# SENTIMENT MODELS
# ===============================
def init_models():
    logging.info("Initializing VADER...")
    vader = SentimentIntensityAnalyzer()
    
    logging.info(f"Loading FinBERT model: {FINBERT_MODEL}")
    finbert = pipeline(
        "sentiment-analysis",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=-1  # -1 = CPU, 0 = GPU if available
    )
    
    return vader, finbert

# ===============================
# VADER SENTIMENT
# ===============================
def get_vader_score(text):
    try:
        score = vader.polarity_scores(text)['compound']
        return score  # Already in [-1, 1]
    except:
        return 0.0

# ===============================
# FINBERT SENTIMENT
# ===============================
def get_finbert_score(texts, finbert):
    try:
        results = finbert(texts)
        scores = []
        for res in results:
            score = res['score']
            if res['label'] == 'negative':
                score = -score
            scores.append(score)
        return scores
    except Exception as e:
        logging.warning(f"FinBERT error on batch: {e}")
        return [0.0] * len(texts)

# ===============================
# MAIN SENTIMENT ANALYSIS
# ===============================
def run_sentiment_analysis():
    if not os.path.exists(INPUT_PATH):
        logging.error(f"Input file not found: {INPUT_PATH}")
        return
    
    # Load data
    df = pd.read_csv(INPUT_PATH)
    logging.info(f"Loaded {len(df)} articles for sentiment analysis")
    
    # Ensure text_nlp is string and clean
    df['text_nlp'] = df['text_nlp'].astype(str).fillna('')
    
    # Initialize models
    vader, finbert = init_models()
    
    # VADER (fast, single pass)
    logging.info("Running VADER sentiment...")
    df['vader_score'] = df['text_nlp'].apply(get_vader_score)
    
    # FinBERT (batched for speed)
    logging.info("Running FinBERT sentiment (batched)...")
    texts = df['text_nlp'].tolist()
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    
    finbert_scores = []
    for i, batch in enumerate(batches):
        if i % 50 == 0:
            logging.info(f"Processing FinBERT batch {i}/{len(batches)}")
        scores = get_finbert_score(batch, finbert)
        finbert_scores.extend(scores)
    
    df['finbert_score'] = finbert_scores
    
    # Ensemble: simple average (you can weight FinBERT more if you want)
    df['sentiment_score'] = (df['vader_score'] + df['finbert_score']) / 2
    
    # Optional: normalize to [-1, 1] just in case
    df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Sentiment analysis completed. Saved to {OUTPUT_PATH}")
    logging.info(f"Final sentiment stats:")
    logging.info(f"  Mean: {df['sentiment_score'].mean():.4f}")
    logging.info(f"  Std:  {df['sentiment_score'].std():.4f}")
    logging.info(f"  Positive: {(df['sentiment_score'] > 0).sum()}")
    logging.info(f"  Negative: {(df['sentiment_score'] < 0).sum()}")
    logging.info(f"  Neutral:  {(df['sentiment_score'] == 0).sum()}")
    
    return df

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    run_sentiment_analysis()