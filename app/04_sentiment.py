# app/04_sentiment.py
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
import torch
from joblib import Parallel, delayed
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = 'data/processed/unified_noticias_nlp_ready.csv'
OUTPUT_DIR = 'data/processed/'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'news_with_sentiment.csv')

# Model configuration
FINBERT_MODEL = 'ProsusAI/finbert'
BATCH_SIZE = 32 
MAX_TEXT_LEN = 500  # To prevent error >512 tokens

# Use GPU if available
DEVICE = 0 if torch.cuda.is_available() else -1
logging.info(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")

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
        device=DEVICE
    )
    
    return vader, finbert

# ===============================
# VADER SENTIMENT
# ===============================
def get_vader_score(text):
    try:
        score = vader.polarity_scores(text)['compound']
        return score
    except:
        return 0.0

# ===============================
# FINBERT SENTIMENT (with truncate)
# ===============================
def truncate_text(text):
    return str(text)[:MAX_TEXT_LEN]

def get_finbert_score(texts, finbert):
    try:
        truncated_texts = [truncate_text(t) for t in texts]
        results = finbert(truncated_texts)
        scores = []
        for res in results:
            label = res['label']
            score = res['score']
            
            # --- CORRECCIÓN LÓGICA ---
            if label == 'positive':
                scores.append(score)
            elif label == 'negative':
                scores.append(-score)
            else: # neutral
                scores.append(0.0) # El neutral DEBE ser 0.0
            # --------------------------
                
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
    if 'text_nlp' not in df.columns:
        df['title'] = df.get('title', '').fillna('').astype(str)
        df['description'] = df.get('description', '').fillna('').astype(str)
        df['text_nlp'] = df['title'] + ' ' + df['description']

    df['text_nlp'] = df['text_nlp'].astype(str).fillna('')
    df = df[df['text_nlp'].str.strip() != ''].copy()
    df.reset_index(drop=True, inplace=True)
    
    # Initialize models
    global vader, finbert
    vader, finbert = init_models()
    
    # VADER (fast)
    logging.info("Running VADER sentiment...")
    df['vader_score'] = df['text_nlp'].apply(get_vader_score)
    
    # FinBERT (batched)
    logging.info("Running FinBERT sentiment (batched with truncate)...")
    texts = df['text_nlp'].tolist()
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    
    finbert_scores = []
    for i, batch in enumerate(batches):
        if i % 50 == 0:
            logging.info(f"Processing FinBERT batch {i}/{len(batches)}")
        scores = get_finbert_score(batch, finbert)
        finbert_scores.extend(scores)
    
    df['finbert_score'] = finbert_scores
    
    # Ensemble: average
    df['sentiment_score'] = (df['finbert_score'] * 0.7) + (df['vader_score'] * 0.3)
    df['sentiment_score'] = df['sentiment_score'].clip(-1, 1)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Sentiment analysis completed. Saved to {OUTPUT_PATH}")
    
    # Final stats
    logging.info("Final sentiment stats:")
    logging.info(f"  Mean sentiment: {df['sentiment_score'].mean():.4f}")
    logging.info(f"  Std sentiment:  {df['sentiment_score'].std():.4f}")
    logging.info(f"  Positive: {(df['sentiment_score'] > 0).sum()} ({(df['sentiment_score'] > 0).mean()*100:.1f}%)")
    logging.info(f"  Negative: {(df['sentiment_score'] < 0).sum()} ({(df['sentiment_score'] < 0).mean()*100:.1f}%)")
    logging.info(f"  Neutral:  {(df['sentiment_score'] == 0).sum()} ({(df['sentiment_score'] == 0).mean()*100:.1f}%)")
    
    return df

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    run_sentiment_analysis()