"""
Funciones auxiliares para cargar datos de precios y noticias desde CSV,
y para generar series OHLC cuando no vienen incluidas.

Este módulo está orientado al front-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from pathlib import Path


@dataclass
class PriceData:
    """
    Contenedor para datos de precios:
    - raw: dataframe original
    - ohlc: dataframe con columnas open/high/low/close (si existen o se calculan)
    - mode: indica si los datos vienen en formato OHLC, si se calcularon,
      o si solo se puede usar una serie lineal.
    """
    raw: pd.DataFrame
    ohlc: pd.DataFrame
    mode: str  # 'ohlc', 'computed_ohlc', 'line'


def _to_utc_datetime(s: pd.Series) -> pd.Series:
    """Convierte una columna a datetime UTC, ignorando errores."""
    return pd.to_datetime(s, utc=True, errors='coerce')


def load_prices(path_or_file) -> pd.DataFrame:
    """
    Carga un CSV de precios.
    Requisitos mínimos:
    - columna timestamp (o date)
    - columna price u OHLC
    """
    df = pd.read_csv(path_or_file)

    # Normalización del nombre de timestamp
    if 'timestamp' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})

    if 'timestamp' not in df.columns:
        raise ValueError("El CSV de precios debe incluir 'timestamp' o 'date'.")

    df['timestamp'] = _to_utc_datetime(df['timestamp'])

    # Normalización de columnas numéricas
    for c in ['open', 'high', 'low', 'close', 'price']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    return df


def build_ohlc(prices_df: pd.DataFrame, interval: str = '1h', min_points_per_candle: int = 2) -> PriceData:
    """
    Construye un dataframe OHLC según las siguientes reglas:

    1) Si ya existen columnas open/high/low/close → se usan directamente.
    2) Si solo existe 'price' y hay suficientes puntos por intervalo →
       se calcula OHLC mediante resampling.
    3) Si no se cumplen las condiciones → se devuelve modo 'line'.
    """
    df = prices_df.copy()

    # Caso 1: OHLC ya disponible
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        ohlc = df[['timestamp', 'open', 'high', 'low', 'close']].dropna(subset=['close']).copy()
        return PriceData(raw=df, ohlc=ohlc, mode='ohlc')

    # Caso 2: solo serie de precios
    if 'price' not in df.columns:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')

    s = df[['timestamp', 'price']].dropna(subset=['price']).copy()
    if s.empty:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')

    s = s.set_index('timestamp')

    # Verificar densidad mínima por intervalo
    counts = s['price'].resample(interval).count()
    if counts.max() < min_points_per_candle:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')

    # Resampling OHLC
    o = s['price'].resample(interval).first()
    h = s['price'].resample(interval).max()
    l = s['price'].resample(interval).min()
    c = s['price'].resample(interval).last()

    ohlc = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna(subset=['close']).reset_index()
    return PriceData(raw=df, ohlc=ohlc, mode='computed_ohlc')


def load_news(path_or_file) -> pd.DataFrame:
    """
    Carga un CSV de noticias.
    Requiere columna 'published_at'.
    Normaliza texto y elimina duplicados comunes.
    """
    df = pd.read_csv(path_or_file)

    if 'published_at' not in df.columns:
        raise ValueError("El CSV de noticias debe incluir 'published_at'.")

    df['published_at'] = _to_utc_datetime(df['published_at'])

    for c in ['title', 'description', 'source', 'axis']:
        if c in df.columns:
            df[c] = df[c].astype('string')

    df = df.dropna(subset=['published_at']).sort_values('published_at')

    # Eliminación de duplicados típicos
    key_cols = [c for c in ['published_at', 'title', 'source'] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols)

    return df


def filter_news_timewindow(news_df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Filtra noticias dentro de un rango temporal."""
    return news_df[(news_df['published_at'] >= start_ts) & (news_df['published_at'] <= end_ts)].copy()


def default_relevance_filter(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtro simple para noticias relacionadas con Bitcoin.
    Busca coincidencias en título y descripción.
    """
    if news_df.empty:
        return news_df

    txt = (news_df.get('title', '').fillna('') + ' ' +
           news_df.get('description', '').fillna('')).str.lower()

    mask = txt.str.contains(r'(bitcoin|btc)', regex=True)
    return news_df[mask].copy()


# Palabras clave para un scoring básico de sentimiento
POS = {"bull", "rally", "gain", "up", "surge", "support", "buy", "approval", "adoption"}
NEG = {"bear", "dump", "down", "selloff", "risk", "regulation", "ban", "hack", "volatility"}


def quick_sentiment_score(news_df: pd.DataFrame) -> Tuple[float, str, int, int]:
    """
    Calcula un puntaje de sentimiento simple basado en palabras clave.
    Devuelve:
    - score numérico
    - etiqueta ('Positivo', 'Negativo', 'Neutral')
    - número de hits positivos
    - número de hits negativos
    """
    if news_df.empty:
        return 0.0, 'Neutral', 0, 0

    txt = (news_df.get('title', '').fillna('') + ' ' +
           news_df.get('description', '').fillna('')).str.lower()

    pos_hits = txt.apply(lambda s: sum(w in s for w in POS)).sum()
    neg_hits = txt.apply(lambda s: sum(w in s for w in NEG)).sum()

    total = max(pos_hits + neg_hits, 1)
    score = (pos_hits - neg_hits) / total

    label = 'Positivo' if score > 0.15 else 'Negativo' if score < -0.15 else 'Neutral'
    return float(score), label, int(pos_hits), int(neg_hits)


def load_prediction_csv(path: str, pred_col_name: str) -> pd.DataFrame:
    """
    Loads a predictions CSV if it exists. Otherwise returns empty DF.
    Normalizes timestamp and renames prediction column to pred_col_name.
    Expected columns: timestamp + (y_pred/forecast/prediction/...) and optional y_true.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)

    # Normalize timestamp column name
    if "timestamp" not in df.columns:
        for alt in ["date", "datetime", "time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "timestamp"})
                break

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Normalize common prediction column names
    rename_map = {
        "y_pred": pred_col_name,
        "pred": pred_col_name,
        "forecast": pred_col_name,
        "prediction": pred_col_name,
        "actual": "y_true",
        "price": "y_true",
        "y": "y_true",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # If the prediction column still isn't present, give up safely
    if pred_col_name not in df.columns:
        return pd.DataFrame()

    keep_cols = ["timestamp", pred_col_name]
    if "y_true" in df.columns:
        keep_cols.append("y_true")

    return df[keep_cols]
