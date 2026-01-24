# app/compute_sentiment.py

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

INPUT_PATH_TEST = 'data/raw/news_raw.csv'
OUTPUT_DIR_TEST = 'data/interim/'
OUTPUT_PATH_TEST = os.path.join(OUTPUT_DIR_TEST, 'news_with_sentiment.csv')

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
            
            if label == 'positive':
                scores.append(score)
            elif label == 'negative':
                scores.append(-score)
            else:  # neutral
                scores.append(0.0)
                
        return scores
    except Exception as e:
        logging.warning(f"FinBERT error on batch: {e}")
        return [0.0] * len(texts)

# ===============================
# MAIN SENTIMENT ANALYSIS
# ===============================
def run_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sentiment on input DataFrame.
    Assumes 'text_nlp' column exists; builds if not.
    Returns df with added sentiment scores (no save).
    """
    if df.empty:
        logging.warning("Empty input DataFrame")
        return df
    
    logging.info(f"Processing {len(df)} articles for sentiment analysis")
    
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
    
    # Final stats (log only, no save)
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
    # This part ONLY runs when you execute this file directly
    if os.path.exists(INPUT_PATH_TEST):
        print("--- STANDALONE TEST START ---")
        df_test = pd.read_csv(INPUT_PATH_TEST)
        
        # Run the modular function
        df_result = run_sentiment_analysis(df_test)
        
        # MANUAL SAVE: Uncomment these 2 lines if you want to save the CSV during your tests
        # os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)
        # df_result.to_csv(OUTPUT_PATH_TEST, index=False)
        
        print(f"Test finished. Articles processed: {len(df_result)}")
        print("--- STANDALONE TEST END ---")
    else:
        print(f"❌ Test file not found at: {INPUT_PATH_TEST}")