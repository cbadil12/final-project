# app/05_aggregate.py
# ===============================
# IMPORTS
# ===============================
import os
import logging

import pandas as pd
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = 'data/processed/news_with_sentiment.csv'
OUTPUT_DIR = 'data/processed/'

WINDOWS = ['1h', '4h']
ROLLING_WINDOW = 7
SHOCK_THRESHOLD = 2.0
MIN_PERIODS_ROLLING = 3

# ===============================
# AGGREGATION FUNCTION
# ===============================
def aggregate_temporal(window):
    try:
        df = pd.read_csv(INPUT_PATH)
        logging.info(f"Starting aggregation for window: {window} ({len(df)} rows loaded)")

        # Prepare datetime index
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
        df = df.sort_values('published_at')
        df.set_index('published_at', inplace=True)

        # Group by time window and axis
        agg = df.groupby([pd.Grouper(freq=window), 'axis']).agg(
            sentiment_mean=('sentiment_score', 'mean'),
            sentiment_count=('sentiment_score', 'count'),
            sentiment_std=('sentiment_score', 'std')
        ).reset_index()

        # Pivot to get wide format per axis
        pivot = agg.pivot(
            index='published_at',
            columns='axis',
            values=['sentiment_mean', 'sentiment_count', 'sentiment_std']
        )
        pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]

        # Total sentiment (all axes combined)
        total = df.groupby(pd.Grouper(freq=window)).agg(
            sentiment_total_mean=('sentiment_score', 'mean'),
            total_count=('sentiment_score', 'count')
        )

        # Combine
        final = pivot.join(total)

        # Handle missing values carefully
        mean_cols = [c for c in final.columns if 'mean' in c]
        count_cols = [c for c in final.columns if 'count' in c]
        std_cols = [c for c in final.columns if 'std' in c]

        # Means: neutral when no news
        final[mean_cols] = final[mean_cols].fillna(0)

        # Counts & stds: NaN when no news (preserve information)
        final[count_cols + std_cols] = final[count_cols + std_cols].where(
            final[count_cols] > 0, np.nan
        )

        # Rolling statistics (only past data)
        original_mean_cols = [c for c in final.columns if 'mean' in c]

        for col in original_mean_cols:
            # Shift to avoid look-ahead
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

            # Shocks based on past rolling stats
            final[f'{col}_shock'] = np.where(
                final[col] > roll_mean + SHOCK_THRESHOLD * roll_std, 1,
                np.where(final[col] < roll_mean - SHOCK_THRESHOLD * roll_std, -1, 0)
            )

            # Simple momentum
            final[f'{col}_momentum'] = final[col].diff()

        # Lags (past values)
        for col in original_mean_cols:
            final[f'{col}_lag1'] = final[col].shift(1)

        # Divergences
        if 'sentiment_mean_BTC' in final.columns and 'sentiment_mean_MACRO' in final.columns:
            final['div_btc_macro'] = final['sentiment_mean_BTC'] - final['sentiment_mean_MACRO']

        # News volume ratio BTC vs MACRO
        if 'sentiment_count_BTC' in final.columns and 'sentiment_count_MACRO' in final.columns:
            final['news_ratio_btc_macro'] = final['sentiment_count_BTC'] / (
                final['sentiment_count_MACRO'] + 1e-6
            )

        # Final cleanup
        final = final.sort_index()
        final = final.loc[~final.index.duplicated(keep='first')]

        # Save
        output_path = os.path.join(OUTPUT_DIR, f'aggregated_{window}.csv')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final.to_csv(output_path)
        logging.info(f"Saved aggregated data for {window} to {output_path} ({len(final)} rows)")

    except Exception as e:
        logging.error(f"Error aggregating for {window}: {e}")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    for w in WINDOWS:
        aggregate_temporal(w)
    logging.info("Temporal aggregation completed for all windows.")