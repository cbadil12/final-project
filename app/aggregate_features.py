# app/aggregate_features.py

# ===============================
# IMPORTS
# ===============================
import logging

import pandas as pd
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Manual test paths
INPUT_PATH = 'data/interim/news_with_sentiment.csv'
FNG_PATH = 'data/raw/fear_greed.csv'
OUTPUT_DIR = 'data/processed/'

ROLLING_WINDOW = 7
SHOCK_THRESHOLD = 2.0
MIN_PERIODS_ROLLING = 3
FNG_DEFAULT = 50

# ===============================
# LOAD F&G HELPER
# ===============================
def load_fng(fng_path: str = "data/raw/fear_greed.csv") -> pd.Series:
    try:
        df = pd.read_csv(fng_path)
        
        # Normalize column names
        if "fear_greed_index" not in df.columns:
            candidates = [c for c in df.columns if "fear" in c.lower() or "greed" in c.lower() or "fng" in c.lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: "fear_greed_index"})
        
        if "datetime" not in df.columns:
            df = df.rename(columns={df.columns[0]: "datetime"})
        
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.floor("H")
        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        
        fng = pd.to_numeric(df["fear_greed_index"], errors="coerce").dropna()
        logging.info(f"Loaded F&G: {len(fng)} rows | {fng.index.min()} -> {fng.index.max()}")
        return fng
    except Exception as e:
        logging.error(f"Error loading F&G: {e}")
        return pd.Series()

# ===============================
# AGGREGATION FUNCTION
# ===============================
def aggregate_features(
    df: pd.DataFrame,
    freq: str = '1H',
    include_fng: bool = True,
    fng_path: str = "data/raw/fear_greed.csv"
) -> pd.DataFrame:
    """
    Aggregates sentiment data by time frequency and axis.
    Input: DF with 'published_at', 'axis', 'sentiment_score'.
    Returns aggregated DF with features (means, counts, stds, rolls, lags, shocks, momentum, div, ratio, optional F&G).
    """
    if df.empty:
        logging.warning("Empty input DF")
        return pd.DataFrame()
    
    try:
        logging.info(f"Aggregating for freq: {freq} ({len(df)} rows input)")
        
        # Prepare datetime index
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
        df = df.sort_values('published_at')
        df.set_index('published_at', inplace=True)

        # Group by time freq and axis
        agg = df.groupby([pd.Grouper(freq=freq), 'axis']).agg(
            sentiment_mean=('sentiment_score', 'mean'),
            sentiment_count=('sentiment_score', 'count'),
            sentiment_std=('sentiment_score', 'std')
        ).reset_index()

        # Pivot to wide format per axis
        pivot = agg.pivot(
            index='published_at',
            columns='axis',
            values=['sentiment_mean', 'sentiment_count', 'sentiment_std']
        )
        pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]

        # Total sentiment (all axes combined)
        total = df.groupby(pd.Grouper(freq=freq)).agg(
            sentiment_total_mean=('sentiment_score', 'mean'),
            total_count=('sentiment_score', 'count')
        )

        # Combine
        final = pivot.join(total)

        # Handle missing values
        mean_cols = [c for c in final.columns if 'mean' in c]
        count_cols = [c for c in final.columns if 'count' in c]
        std_cols = [c for c in final.columns if 'std' in c]

        final[mean_cols] = final[mean_cols].fillna(0)
        final[count_cols + std_cols] = final[count_cols + std_cols].where(
            final[count_cols] > 0, np.nan
        )

        # Rolling statistics (only past data)
        original_mean_cols = [c for c in final.columns if 'mean' in c]

        for col in original_mean_cols:
            roll_mean = final[col].shift(1).rolling(
                window=ROLLING_WINDOW,
                min_periods=MIN_PERIODS_ROLLING
            ).mean()

            roll_std = final[col].shift(1).rolling(
                window=ROLLING_WINDOW,
                min_periods=MIN_PERIODS_ROLLING
            ).std()

            final[f'{col}_roll_mean'] = roll_mean
            final[f'{col}_roll_std'] = roll_std

            final[f'{col}_shock'] = np.where(
                final[col] > roll_mean + SHOCK_THRESHOLD * roll_std, 1,
                np.where(final[col] < roll_mean - SHOCK_THRESHOLD * roll_std, -1, 0)
            )

            final[f'{col}_momentum'] = final[col].diff()

        # Lags
        for col in original_mean_cols:
            final[f'{col}_lag1'] = final[col].shift(1)

        # Divergences
        if 'sentiment_mean_BTC' in final.columns and 'sentiment_mean_MACRO' in final.columns:
            final['div_btc_macro'] = final['sentiment_mean_BTC'] - final['sentiment_mean_MACRO']

        # News ratio
        if 'sentiment_count_BTC' in final.columns and 'sentiment_count_MACRO' in final.columns:
            final['news_ratio_btc_macro'] = final['sentiment_count_BTC'] / (
                final['sentiment_count_MACRO'] + 1e-6
            )

        # Add F&G if requested
        if include_fng:
            fng = load_fng(fng_path)
            if not fng.empty:
                final['fear_greed_index'] = fng.reindex(final.index).ffill().fillna(FNG_DEFAULT)
            else:
                final['fear_greed_index'] = FNG_DEFAULT
                logging.warning("F&G not loaded - using default")

        # Final cleanup
        final = final.sort_index()
        final = final.loc[~final.index.duplicated(keep='first')]

        logging.info(f"Aggregation completed: {len(final)} rows")

        return final
    except Exception as e:
        logging.exception(f"Error aggregating features for freq={freq}: {e}")
        return pd.DataFrame()
# ===============================
# ENTRY POINT (for test/static)
# ===============================
if __name__ == "__main__":
    import os
    
    # 1. Load data using the paths defined in CONFIGURATION
    if os.path.exists(INPUT_PATH):
        logging.info(f"--- STARTING STANDALONE TEST FROM: {INPUT_PATH} ---")
        input_df = pd.read_csv(INPUT_PATH)
        
        # 2. Loop through your desired frequencies
        for freq in ['1H', '4H']:
            # Run the aggregation logic
            aggregated = aggregate_features(
                input_df, 
                freq=freq, 
                include_fng=True, 
                fng_path=FNG_PATH
            )
            
            # 3. MANUAL SAVE SECTION
            # To save the files to 'data/processed/', just uncomment the lines below:
            
            # output_filename = f'aggregated_{freq.lower()}.csv'
            # output_path = os.path.join(OUTPUT_DIR, output_filename)
            # os.makedirs(OUTPUT_DIR, exist_ok=True)
            # aggregated.to_csv(output_path)
            # logging.info(f"✅ File saved manually to: {output_path}")

        logging.info("--- STANDALONE TEST FINISHED ---")
    else:
        logging.error(f"❌ Input file not found at {INPUT_PATH}. Please run the sentiment script first.")