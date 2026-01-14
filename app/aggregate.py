# app/aggregate.py
# ===============================
# IMPORTS
# ===============================
# 1. Standard library
import os
import logging

# 2. Third-party libraries
import pandas as pd
import numpy as np

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = 'data/processed/news_with_sentiment.csv'
OUTPUT_DIR = 'data/processed/'

WINDOWS = ['4h', '1h']  # Usamos 'h' para evitar warning de depreciación
ROLLING_WINDOW = 7
SHOCK_THRESHOLD = 2

# ===============================
# AGGREGATION FUNCTION
# ===============================
def aggregate_temporal(window):
    try:
        df = pd.read_csv(INPUT_PATH)
        logging.info(f"Aggregando para ventana: {window} (cargadas {len(df)} filas)")
        
        # Convertir fecha y poner índice
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
        df.set_index('published_at', inplace=True)
        
        # Agrupar por ventana y axis
        agg = df.groupby([pd.Grouper(freq=window), 'axis']).agg(
            sentiment_mean=('sentiment_score', 'mean'),
            sentiment_count=('sentiment_score', 'count'),
            sentiment_std=('sentiment_score', 'std')
        ).reset_index()
        
        # Pivot para columnas por axis
        pivot = agg.pivot(index='published_at', columns='axis', values=['sentiment_mean', 'sentiment_count', 'sentiment_std'])
        pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]
        
        # Total general
        total = df.groupby(pd.Grouper(freq=window)).agg(
            sentiment_total=('sentiment_score', 'mean'),
            total_count=('sentiment_score', 'count')
        )
        
        # Unir
        final = pivot.join(total)
        final = final.fillna(0)  # No noticias = neutral
        
        # Columnas originales para rolling/lags/shocks (evita redundancias)
        original_mean_cols = [c for c in final.columns if 'sentiment_mean' in c or 'sentiment_total' in c]
        
        # Rolling stats
        for col in original_mean_cols:
            final[f'{col}_roll_mean'] = final[col].rolling(ROLLING_WINDOW, min_periods=1).mean()
            final[f'{col}_roll_std'] = final[col].rolling(ROLLING_WINDOW, min_periods=1).std()
        
        # Lags
        for col in original_mean_cols:
            final[f'{col}_lag1'] = final[col].shift(1)
        
        # Shocks
        for col in original_mean_cols:
            roll_mean = f'{col}_roll_mean'
            roll_std = f'{col}_roll_std'
            final[f'{col}_shock'] = np.where(
                final[col] > final[roll_mean] + SHOCK_THRESHOLD * final[roll_std], 1,
                np.where(final[col] < final[roll_mean] - SHOCK_THRESHOLD * final[roll_std], -1, 0)
            )
        
        # Divergencias
        if 'sentiment_mean_BTC' in final.columns and 'sentiment_mean_MACRO' in final.columns:
            final['div_btc_macro'] = final['sentiment_mean_BTC'] - final['sentiment_mean_MACRO']
        
        # Guardar
        output_path = os.path.join(OUTPUT_DIR, f'aggregated_{window}.csv')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final.to_csv(output_path)
        logging.info(f"Guardado {output_path} con {len(final)} filas")
        
    except Exception as e:
        logging.error(f"Error en agregación para {window}: {e}")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    for w in WINDOWS:
        aggregate_temporal(w)
    logging.info("Agregación completada para todas las ventanas.")