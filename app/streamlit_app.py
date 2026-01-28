# ===============================================================
# 📊 BITCOIN PREDICTOR PPLICATION
# ===============================================================
# Author: Carlos Badillo,Mercedes Salaverri, Adrian
# Created on: January 2026
# Description:
#     This application provides an interactive, web-based environment
#     for performing an Bitcoin price prediction automatically.
#     All within a simple Streamlit interface.
# -*- coding: utf-8 -*-

# ===============================
# IMPORTS
# ===============================
# 1. Standard library
from __future__ import annotations
from datetime import datetime, timezone, time as dtime
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
# 2. Third-party libraries (from requirements.txt)
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import altair as alt
# 3. Modules imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from app.styles import set_styles, ICONS
from app.text_content import TEXT_TITLE, TEXT_DISCLAIMER, TEXT_SUGGESTIONS, TEXT_CONTEXT, TEXT_EDA_1, TEXT_EDA_2, TEXT_EDA_SENTIMENT
# 4. Dynamic imports
DYN_IMPORT_ERROR = None
run_dynamic_predict = None
run_fused_predict = None
try:
    from app.dynamic_predict import run_dynamic_predict
except Exception as e:
    run_dynamic_predict = None
    DYN_IMPORT_ERROR = f"import run_dynamic_predict failed: {repr(e)}"

# Import opcional: si no existe, NO rompe el resto
try:
    from app.dynamic_predict import run_fused_predict
except Exception as e:
    run_fused_predict = None
    if DYN_IMPORT_ERROR:
        DYN_IMPORT_ERROR += f" | import run_fused_predict failed: {repr(e)}"
    else:
        DYN_IMPORT_ERROR = f"import run_fused_predict failed: {repr(e)}"

# ===============================
# CLASSES
# ===============================
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

# ===============================
# PATHS
# ===============================
# Root folder
ROOT_PATH = Path(ROOT)
# Project folders
ASSETS_DIR = ROOT_PATH / "assets"
RAW_DIR = ROOT_PATH / "data" / "raw"
PROCESSED_DIR = ROOT_PATH / "data" / "processed"
# Raw data
PRICE_1H_RAW_PATH   = RAW_DIR / "btcusd-1h.csv"
PRICE_4H_RAW_PATH   = RAW_DIR / "btcusd-4h.csv"
PRICE_24H_RAW_PATH  = RAW_DIR / "btcusd-24h.csv"
NEWS_RAW_PATH       = RAW_DIR / "news_raw.csv"
FNG_RAW_PATH        = RAW_DIR / "fear_greed.csv"
# Processed data con features + targets for sentiment models)
DATASET_SENT_1H_PATH = PROCESSED_DIR / "dataset_sentiment_target_1h.csv"
DATASET_SENT_4H_PATH = PROCESSED_DIR / "dataset_sentiment_target_4h.csv"
# predictions (outputs sentimiento)
PRED_RF_1H_PATH  = PROCESSED_DIR / "predictions_rf_clf_1h.csv"
PRED_RF_4H_PATH  = PROCESSED_DIR / "predictions_rf_clf_4h.csv"
PRED_XGB_1H_PATH = PROCESSED_DIR / "predictions_xgb_clf_1h.csv"
PRED_XGB_4H_PATH = PROCESSED_DIR / "predictions_xgb_clf_4h.csv"
# (opcional) dataset only sentiment”
SENTIMENT_ONLY_1H_PATH = PROCESSED_DIR / "sentiment_only_1h.csv"

# ===============================
# STREAMLIT CONFIGURATION
# ===============================
NAV_INDEX = {"Home": 0, "Dashboard": 1, "Overview": 2} # Nvaigation menu
st.set_page_config(page_title='BTC Predictor', page_icon='📈', layout='wide') # Page
st.markdown(ICONS, unsafe_allow_html=True) # Bootstrap icons
st.markdown(set_styles(ASSETS_DIR / "btc_logo_gray.png"), unsafe_allow_html=True) # Global styles

# ===============================
# STREAMLIT FUNCTIONS
# ===============================
# Navigates through menu
def go_to(page_name: str):
    st.session_state["nav_manual_select"] = NAV_INDEX.get(page_name, 0)

# Renderizes disclaimer message
def render_disclaimer():
    st.markdown(f"<div class='disclaimer'>{TEXT_DISCLAIMER}</div>", unsafe_allow_html=True)

# Builds OHLC graph using Altair
def candlestick_chart(df_ohlc: pd.DataFrame,
                      ts_col: str = 'datetime',
                      max_candle_no: int = 500):
    df = df_ohlc.copy()
    # 1. Normalize timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    df = df.dropna(subset=[ts_col])
    # 2. Delete EXACT duplicates
    df = df.drop_duplicates(subset=[ts_col])
    # 3. Sort
    df = df.sort_values(ts_col)
    # 4. Limit candle number to show
    if len(df) > max_candle_no:
        df = df.tail(max_candle_no)
    # 5. Build chart
    base = alt.Chart(df).encode(x=f"{ts_col}:T")
    # High–Low vertical line
    rule = base.mark_rule(color='white', size=1.2).encode(y=alt.Y('low:Q', scale=alt.Scale(zero=False)), y2='high:Q')
    # Candle body
    bar = base.mark_bar(size=4).encode(
        y='open:Q',
        y2='close:Q',
        color=alt.condition('datum.open <= datum.close', alt.value('#22a884'), alt.value("#b91e0d")),
        tooltip=[f"{ts_col}:T", 'open:Q', 'high:Q', 'low:Q', 'close:Q']
    )
    # Compose chart
    chart = (rule + bar).properties(width='container').configure_view(strokeWidth=0).configure_axis(
        gridColor='#333333',
        labelColor='white',
        titleColor='white'
    )
    return chart

# Generates moving average (MA) line-graph
# - price_col: columna de precios
#- ma_window: tamaño de la ventana para la media móvil
def line_with_ma(df: pd.DataFrame,
                 ts_col: str = 'datetime',
                 price_col: str = 'price',
                 ma_window: int = 24):
    d = df[[ts_col, price_col]].dropna().copy()
    d = d.sort_values(ts_col)
    d['ma'] = d[price_col].rolling(ma_window, min_periods=1).mean()
    base = alt.Chart(d).encode(x=f'{ts_col}:T')
    line = base.mark_line(color='#1f77b4').encode(
        y=f'{price_col}:Q',
        tooltip=[f'{ts_col}:T', f'{price_col}:Q']
    )
    area = base.mark_area(opacity=0.2, color='#1f77b4').encode( y=f'{price_col}:Q' )
    ma = base.mark_line(color='#fde725').encode(y='ma:Q')
    return area + line + ma

# ===============================
# HELPER FUNCTIONS
# ===============================
# Builds an OHLC dataframe using the following rules:
# 1) If open/high/low/close columns already exist → use them directly
# 2) If only 'price' exists and there are enough points per interval → compute OHLC via resampling.
# 3) If conditions are not met → return mode 'line'
def build_ohlc(prices_df: pd.DataFrame, interval: str = '1h', min_points_per_candle: int = 2) -> PriceData:
    df = prices_df.copy()
    _freq_map = {
        "1h": "1H",
        "4h": "4H",
        "24h": "24H",
        "1d": "1D",
        "d": "1D",
    }
    freq = _freq_map.get(str(interval).lower(), interval)
    # Case 1: OHLC already avaiable
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors='coerce')
            ohlc = df[['datetime', 'open', 'high', 'low', 'close']].dropna(subset=['close']).copy()
        else:
            ohlc = df[['open', 'high', 'low', 'close']].dropna(subset=['close']).copy()
        return PriceData(raw=df, ohlc=ohlc, mode='ohlc')
    # Caso 2: only price series
    if 'price' not in df.columns:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')
    if "datetime" not in df.columns:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors='coerce')    
    s = df[['datetime', 'price']].dropna(subset=['price']).copy()
    if s.empty:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')
    s = s.set_index("datetime").sort_index()
    # Secure datetime index UTC
    s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    s = s.dropna(subset=["price"])
    # Verifiy min density per interval
    counts = s['price'].resample(freq).count()
    valid_ratio = float((counts >= min_points_per_candle).mean()) if len(counts) else 0.0
    if valid_ratio < 0.60:
        return PriceData(raw=df, ohlc=pd.DataFrame(), mode='line')
    # Resampling OHLC
    o = s['price'].resample(freq).first()
    h = s['price'].resample(freq).max()
    l = s['price'].resample(freq).min()
    c = s['price'].resample(freq).last()
    ohlc = pd.DataFrame({'open': o, 'high': h, 'low': l, 'close': c}).dropna(subset=['close']).reset_index()
    return PriceData(raw=df, ohlc=ohlc, mode='computed_ohlc')

# Find the Nearest Row by datetime
def _nearest_row_by_ts(df: pd.DataFrame,
                       target_ts: pd.Timestamp,
                       ts_col="datetime"):
    if df is None or df.empty or ts_col not in df.columns or target_ts is None:
        return None
    tmp = df.dropna(subset=[ts_col]).copy()
    if tmp.empty:
        return None
    tmp[ts_col] = pd.to_datetime(tmp[ts_col], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=[ts_col]).sort_values(ts_col)
    pos = tmp[ts_col].searchsorted(target_ts)
    if pos <= 0:
        return tmp.iloc[0]
    if pos >= len(tmp):
        return tmp.iloc[-1]
    before = tmp.iloc[pos - 1]
    after = tmp.iloc[pos]
    return before if abs((before[ts_col] - target_ts).total_seconds()) <= abs((after[ts_col] - target_ts).total_seconds()) else after

def load_pred_if_exists(model_name: str,
                        resolution: str,
                        y_col: str = "y_pred") -> pd.DataFrame:
    p = PROCESSED_DIR / f"predictions_{model_name}_{resolution}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = load_prediction_csv(str(p), y_col)
    if df is None or df.empty:
        return pd.DataFrame()
    # Normaliza datetime como columna
    if "datetime" not in df.columns:
        # Caso: datetime viene como índice
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "datetime"})
        # Caso alternativo: columna timestamp
        elif "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
    # Fuerza tipo datetime UTC si existe
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df

@st.cache_data(show_spinner=False)
# Loads a news CSV.
# - Requires a 'published_at' column.
# - Normalizes text and removes common duplicates.
def load_news(path_or_file: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_file)
    if 'published_at' not in df.columns:
        raise ValueError("El CSV de noticias debe incluir 'published_at'.")
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce')
    for c in ['title', 'description', 'source', 'axis']:
        if c in df.columns:
            df[c] = df[c].astype('string')
    df = df.dropna(subset=['published_at']).sort_values('published_at')
    # Erase duplicates
    key_cols = [c for c in ['published_at', 'title', 'source'] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols)
    return df

@st.cache_data(show_spinner=False)
# Loads a prices CSV.
# Minimum requirements:
# - timestamp column (or date)
# - price column or OHLC columns
def _load_prices_compat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # Normalize datetime column name
    if 'datetime' not in df.columns and 'date' in df.columns:
        df = df.rename(columns={'date': 'datetime'})
    if 'datetime' not in df.columns:
        raise ValueError("El CSV de precios debe incluir 'datetime'.")
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    df["price"] = pd.to_numeric(df.get("close"), errors="coerce")
    # Normalize numeric columns
    for c in ['open', 'high', 'low', 'close', 'price']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime')
    return df

# Loads a predictions CSV if it exists; otherwise returns an empty DataFrame.
# Normalizes the timestamp column and renames the prediction column to pred_col_name.
# Expected columns: timestamp + (y_pred/forecast/prediction/...) and optional y_true.
def load_prediction_csv(path: str, pred_col_name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Normalize timestamp column name
    if "datetime" not in df.columns:
        for alt in ["date", "timestamp", "time"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "datetime"})
                break
    if "datetime" not in df.columns:
        return pd.DataFrame()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
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
    keep_cols = ["datetime", pred_col_name]
    if "y_true" in df.columns:
        keep_cols.append("y_true")
    return df[keep_cols]

# Computes a simple keyword-based sentiment score.
# Returns:
# - numeric score
# - label ('Positive', 'Negative', 'Neutral')
# - number of positive hits
# - number of negative hits
def quick_sentiment_score(news_df: pd.DataFrame) -> Tuple[float, str, int, int]:
    if news_df.empty:
        return 0.0, 'Neutral', 0, 0
    # Keywords for a basic sentiment scoring
    POS = {"bull", "rally", "gain", "up", "surge", "support", "buy", "approval", "adoption"}
    NEG = {"bear", "dump", "down", "selloff", "risk", "regulation", "ban", "hack", "volatility"}
    txt = (news_df.get('title', '').fillna('') + ' ' +
           news_df.get('description', '').fillna('')).str.lower()
    pos_hits = txt.apply(lambda s: sum(w in s for w in POS)).sum()
    neg_hits = txt.apply(lambda s: sum(w in s for w in NEG)).sum()
    total = max(pos_hits + neg_hits, 1)
    score = (pos_hits - neg_hits) / total
    label = 'Positivo' if score > 0.15 else 'Negativo' if score < -0.15 else 'Neutral'
    return float(score), label, int(pos_hits), int(neg_hits)

@st.cache_data(show_spinner=False)
def _run_dynamic_predict_cached(
    target_ts_str: str,
    resolution: str,
    task: str = "sentiment",
    mode: str = "auto",
    window_hours: int = 24,
    model_name: str = "xgb_clf"
):
    if run_dynamic_predict is None:
        return {"msg": "dynamic_predict no disponible", "prediction": None, "confidence": 0.0}
    return run_dynamic_predict(
        target_ts=target_ts_str,
        resolution=resolution,
        task=task,
        mode=mode,
        window_hours=window_hours,
        model_name=model_name
    )

@st.cache_data(show_spinner=False)
def _run_fused_predict_cached(
    target_ts_sent_str: str,
    target_ts_price_str: str,
    resolution: str,
    mode: str = "auto",
    window_hours: int = 24,
    sentiment_model_name: str | None = None,
    price_model_name: str | None = None,
):
    # Sentiment NO soporta 24h en tus modelos; degradamos a 4h automáticamente
    sentiment_resolution = resolution if resolution in ("1h", "4h") else "4h"
    price_resolution = resolution
    # Si no tenemos NI dynamic, no hay nada que hacer
    if run_dynamic_predict is None and run_fused_predict is None:
        return {"msg": "dynamic_predict no disponible", "sentiment": None, "price": None}
    # Preferir fused si existe
    if run_fused_predict is not None and resolution != "24h":
        return run_fused_predict(
            target_ts_sent=target_ts_sent_str,
            target_ts_price=target_ts_price_str,
            resolution=resolution,
            mode=mode,
            window_hours=window_hours,
            sentiment_model_name=sentiment_model_name,
            price_model_name=price_model_name,
        )
    # ✅ Fallback: si fused no existe, hacer 2 llamadas como antes (pero dentro de UNA función cacheada)
    sent = run_dynamic_predict(
        target_ts=target_ts_sent_str,
        resolution=sentiment_resolution,
        task="sentiment",
        mode=mode,
        window_hours=window_hours,
        model_name=sentiment_model_name,  # None => default en run_model
    )
    price = run_dynamic_predict(
        target_ts=target_ts_price_str,
        resolution=price_resolution,
        task="price",
        mode=mode,
        window_hours=window_hours,
        model_name=price_model_name,      # None => default en run_model
    )
    return {"msg": "fallback(dynamic x2)", "sentiment": sent, "price": price}

def _direction_from_pred(base: float | None, pred: float | None, neutral_pct: float = 0.002):
    """Dirección usando cambio relativo vs base.
    neutral_pct=0.002 => +-0.2% se considera neutral (evita ruido).
    """
    if base is None or pred is None or base == 0:
        return ("ESCENARIO NEUTRAL", "gray", None)

    delta = (pred - base) / base
    if delta > neutral_pct:
        return ("SUBIDA", "green", delta)
    if delta < -neutral_pct:
        return ("BAJADA", "red", delta)
    return ("ESCENARIO NEUTRAL", "gray", delta)

def _get_base_price_for_target(price_pack, fallback_df, target_ts: pd.Timestamp, price_horizon_hours: int):
    """Intenta base_ref = close en (target_ts - horizon). Si no existe, usa último close <= target_ts."""
    # Preferimos OHLC si existe
    ohlc = getattr(price_pack, "ohlc", None)
    if ohlc is not None and not ohlc.empty and "datetime" in ohlc.columns:
        ohlc = ohlc.sort_values("datetime")
        t_prev = target_ts - pd.Timedelta(hours=price_horizon_hours)
        exact = ohlc[ohlc["datetime"] == t_prev]
        if not exact.empty and "close" in exact.columns:
            return float(exact["close"].iloc[0])

        prev = ohlc[ohlc["datetime"] <= target_ts].tail(1)
        if not prev.empty and "close" in prev.columns:
            return float(prev["close"].iloc[0])
    # Fallback: usar price_window/df original
    if fallback_df is not None and not fallback_df.empty and "datetime" in fallback_df.columns and "price" in fallback_df.columns:
        tmp = fallback_df.dropna(subset=["price"]).sort_values("datetime")
        prev = tmp[tmp["datetime"] <= target_ts].tail(1)
        if not prev.empty:
            return float(prev["price"].iloc[0])
    return None

def combine_final_signal(price_pred: int | None, sentiment_label: str | None):
    """
    Regla final (la que definiste):
    - Precio SUBIDA + Sentimiento Positivo/Neutral => SUBIDA
    - Precio BAJADA + Sentimiento Negativo/Neutral => BAJADA
    - Si chocan => ESCENARIO MIXTO
    - Si falta uno => usa el que esté disponible, si no => NEUTRAL
    """
    if sentiment_label == "Positivo":
        sent_cls = 1
    elif sentiment_label == "Negativo":
        sent_cls = 0
    else:
        sent_cls = None  # Neutral o desconocido
    # Si hay precio, la señal base la decide precio salvo que choque con sentimiento
    if price_pred in (0, 1):
        if sent_cls is None:
            # Neutral => acompaña al precio
            return ("SUBIDA", "green") if price_pred == 1 else ("BAJADA", "red")
        if price_pred == sent_cls:
            return ("SUBIDA", "green") if price_pred == 1 else ("BAJADA", "red")
        return ("ESCENARIO MIXTO", "gray")
    # Si no hay precio, usa sentimiento si es Pos/Neg
    if sent_cls in (0, 1):
        return ("SUBIDA", "green") if sent_cls == 1 else ("BAJADA", "red")
    return ("ESCENARIO NEUTRAL", "gray")

def render_price_block(price_result: dict | None, horizon_label: str, debug_mode: bool):
    """Muestra la predicción técnica (precio) de forma consistente."""
    st.subheader("Predicción técnica (precio)")
    if isinstance(price_result, dict) and price_result.get("prediction") is not None:
        pr_cls = price_result.get("prediction")
        pr_conf = price_result.get("confidence", None)

        label = "SUBIDA" if pr_cls == 1 else "BAJADA"
        pr_color = "green" if pr_cls == 1 else "red"

        st.markdown(
            f"<h4 style='color:{pr_color}; margin:0'>{label} ({horizon_label})</h4>",
            unsafe_allow_html=True
        )
        try:
            pr_prog = int(max(0, min(100, float(pr_conf) * 100))) if pr_conf is not None else 50
        except Exception:
            pr_prog = 50
        st.progress(pr_prog)
        if "proba_up" in price_result:
            try:
                st.caption(f"proba_up: {float(price_result['proba_up']):.2f}")
            except Exception:
                pass
        if debug_mode:
            with st.expander("Detalles precio (debug)", expanded=False):
                st.write(price_result)
    else:
        st.info("Pulsa 🚀 Ejecutar Predicción para calcular la predicción de precio.")

# ===============================
# NAVIGATION MENU RENDERS
# ===============================
def render_home():
    left, center_msg, right = st.columns([1,4,1])
    with center_msg:
        st.markdown(TEXT_TITLE, unsafe_allow_html=True)
    left, center_button, right = st.columns([5,2,5])
    with center_button:
        st.button("Ir a la App →", type="primary", on_click=go_to, args=("Dashboard",))
    left, center_disclaimer, right = st.columns([1,4,1])
    with center_disclaimer:
        render_disclaimer()

def render_overview():
    tab_context, tab_eda_1, tab_eda_2, tab_eda_sentiment,tab_future = st.tabs(["Contexto", "Time-series EDA (Global vs Regímenes)", "Time-series EDA (Resampling)", "EDA sentimientos","Futuras implementaciones"])
    with tab_context:
        st.markdown(TEXT_CONTEXT)
        st.image("assets/pipeline.png", caption="Pipeline", use_container_width=True)
    with tab_eda_1:
        st.markdown(TEXT_EDA_1)
    with tab_eda_2:
        st.markdown(TEXT_EDA_2)
    with tab_eda_sentiment:
        st.markdown(TEXT_EDA_SENTIMENT)
    with tab_future:
        st.markdown(TEXT_SUGGESTIONS)
    st.markdown("</div></div>", unsafe_allow_html=True)
    render_disclaimer()


def render_dashboard():
    # Time in which the dashbord has been opened
    if "opened_at_utc" not in st.session_state:
        st.session_state["opened_at_utc"] = datetime.now(timezone.utc)
    # -------------------------------    
    # Sidebar title
    # -------------------------------
    st.sidebar.title("Panel de control")
    # -------------------------------    
    # Debug toggle
    # -------------------------------
    debug_mode = st.sidebar.toggle("Mostrar detalles técnicos (debug)", value=False)
    if debug_mode and DYN_IMPORT_ERROR:
        st.sidebar.error(f"Error import dynamic_predict: {DYN_IMPORT_ERROR}")
        st.sidebar.write({
            "run_dynamic_predict_is_none": run_dynamic_predict is None,
            "DYN_IMPORT_ERROR": DYN_IMPORT_ERROR
        })
    # -------------------------------    
    # Granularity/Interval selectbox
    # -------------------------------
    interval_options = ["1h", "4h", "24h"]
    interval = st.sidebar.selectbox(
        "Granularidad",
        interval_options,
        index=interval_options.index(st.session_state.get("interval", "1h")) if st.session_state.get("interval", "1h") in interval_options else 0,
        key="interval",
        help="Frecuencia del precio para gráfico y predicción."
    )
    INTERVAL_TO_HOURS = {"1h": 1, "4h": 4, "24h": 24}
    price_horizon_hours = INTERVAL_TO_HOURS.get(interval, 1)
    st.sidebar.caption(f"Predicción: **t+1** (siguiente {interval} → +{price_horizon_hours}h)")
    # -------------------------------    
    # Dataset loading
    # -------------------------------
    # Set raw dataset path according to granularity
    if interval == "1h":
        price_path = PRICE_1H_RAW_PATH
    elif interval == "4h":
        price_path = PRICE_4H_RAW_PATH
    else:
        price_path = PRICE_24H_RAW_PATH
    news_path = NEWS_RAW_PATH
    # Load raw datasets
    prices_df = _load_prices_compat(str(price_path))
    news_df   = load_news(str(news_path)) if news_path.exists() else pd.DataFrame()
    if prices_df.empty:
        st.error("No se pudieron cargar precios (revisa data/raw y/o data/processed).")
        return
    # -------------------------------    
    # Start and End date inputs
    # -------------------------------
    if not news_df.empty:
        max_ts = max(prices_df["datetime"].max(), news_df["published_at"].max())
    else:
        max_ts = prices_df["datetime"].max()
    start_date = st.sidebar.date_input("Fecha inicio", value=st.session_state.get("start_date", (max_ts - pd.Timedelta(days=60)).date()), key="start_date")
    end_date = st.sidebar.date_input("Fecha fin", value=st.session_state.get("end_date", max_ts.date()), key="end_date")
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts   = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    # -------------------------------    
    # Filter relevant news checkbox
    # -------------------------------
    relevance = st.sidebar.checkbox("Filtrar noticias relevantes", value=st.session_state.get("relevance", True), key="relevance", help="Aplica filtro simple ‘bitcoin|btc’ en título/descripcion.")   
    # Build price window
    price_window = prices_df[(prices_df["datetime"] >= start_ts) & (prices_df["datetime"] <= end_ts)].copy()  
    # Build news window
    if not news_df.empty:
        news_window = news_df[(news_df['published_at'] >= start_ts) & (news_df['published_at'] <= end_ts)].copy()
        if relevance:
            if not news_window.empty:
                txt = (news_window.get('title', '').fillna('') + ' ' + news_window.get('description', '').fillna('')).str.lower()
                # Non-capturing group avoids pandas warning about match groups.
                mask = txt.str.contains(r'(?:bitcoin|btc)', regex=True)
                news_window =  news_window[mask].copy()
    else:
        news_window = pd.DataFrame()
    # -------------------------------    
    # Sidebar footer
    # -------------------------------
    st.sidebar.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="meta"><b>Hora de comienzo de esta sesión:</b> {st.session_state["opened_at_utc"]:%Y-%m-%d %H:%M} UTC</div>
            <div class="meta"><b>Datos disponibles hasta:</b> {prices_df['datetime'].max():%Y-%m-%d %H:%M} UTC</div>
            <div class="meta"><b></b></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # -------------------------------    
    # Refresh data button
    # -------------------------------
    refresh_data = st.sidebar.button("Actualizar datos", type="secondary", help="Refresca la información si se actualizaron CSVs en data/processed (limpia caché y recarga).")
    if refresh_data:
        st.cache_data.clear()
        st.rerun()
    st.sidebar.markdown("---")
    # -------------------------------    
    # Prediction button
    # -------------------------------
    run_pred = st.sidebar.button("🚀 Ejecutar Predicción", type="primary", help="Ejecuta la predicción live usando los modelos disponibles.")
    if run_pred:
        st.toast("Ejecutando predicción dinámica…", icon="🚀")
    # -------------------------------    
    # Sidebar disclaimer
    # -------------------------------
    st.sidebar.markdown(
        f"""
        <div class="sidebar-footer">
            <div class="disclaimer">{TEXT_DISCLAIMER}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # ------------------ Precio (OHLC o línea) ------------------
    price_pack = build_ohlc(price_window, interval=interval, min_points_per_candle=2)
    
    st.subheader("Precio histórico de BTC (velas de " + interval + ")")
    max_candle_no = st.number_input(
        "Número máximo de velas a mostrar",
        min_value=500,
        key="max_candle_number",
        max_value=100000, step=500)
    if price_pack.mode in ("ohlc", "computed_ohlc") and not price_pack.ohlc.empty:
        chart = candlestick_chart(price_pack.ohlc, ts_col="datetime", max_candle_no=max_candle_no).properties(height=380)
        st.altair_chart(chart, width='stretch')
    else:
        non_null = price_window.dropna(subset=["price"]).copy()
        if not non_null.empty:
            chart = line_with_ma(non_null, ts_col="datetime", price_col="price", ma_window=24).properties(height=380)
            st.altair_chart(chart, width='stretch')
        else:
            st.warning("No hay datos de precio disponibles en el rango/regimen mostrado.")

    st.divider()

    # ------------------ Panel Sentimiento / Señal ------------------
    col_market_feel, col_prediction, col_final_signal = st.columns([1,1,1])
    RECENT_HOURS = 48
    if news_window.empty:
        sentiment_score, sentiment_label, pos_hits, neg_hits = 0.0, "Neutral", 0, 0
        used_scope = "sin noticias"
    else:
        cutoff = end_ts - pd.Timedelta(hours=RECENT_HOURS)
        news_recent = news_window[news_window["published_at"] >= cutoff].copy()
        used_scope = f"últimas {RECENT_HOURS}h" if not news_recent.empty else f"rango completo (no hubo noticias en últimas {RECENT_HOURS}h)"
        if news_recent.empty:
            news_recent = news_window

        if "sentiment_score" in news_recent.columns:
            s = pd.to_numeric(news_recent["sentiment_score"], errors="coerce").dropna()
            sentiment_score = float(s.mean()) if not s.empty else 0.0
            pos_hits = int((s > 0.05).sum())
            neg_hits = int((s < -0.05).sum())

            # Etiqueta con umbral moderado (evitar falsas neutras)
            if sentiment_score > 0.05:
                sentiment_label = "Positivo"
            elif sentiment_score < -0.05:
                sentiment_label = "Negativo"
            else:
                sentiment_label = "Neutral"
        else:
            sentiment_score, sentiment_label, pos_hits, neg_hits = quick_sentiment_score(news_recent)

    prob = int((sentiment_score + 1) * 50)
    prob = max(0, min(100, prob))
    with col_market_feel:
        st.subheader("Sentimiento del mercado")
        st.metric("Sentimiento agregado", sentiment_label, f"{sentiment_score:+.2f}")
        st.progress(prob)
        st.caption(f"Probabilidad (escala interna): {prob}%")
        st.caption(f"Hits → Positivos: {pos_hits} | Negativos: {neg_hits}")
        st.caption(f"Ventana usada para el sentimiento: {used_scope}")

    # --- Señal final (Global) basada en predicción t+1 ---
    last_series = price_window.dropna(subset=["price"])
    last_price = float(last_series.iloc[-1]["price"]) if not last_series.empty else None
    last_seen_ts = last_series["datetime"].max() if not last_series.empty else None
    
    target_ts_g = None
    if last_seen_ts is not None:
        target_ts_g = last_seen_ts + pd.Timedelta(hours=price_horizon_hours)

    # Cargar predicciones
    if interval == "1h":
        rf_path = PRED_RF_1H_PATH
    else:
        rf_path = PRED_RF_4H_PATH
    
    rf_df_sig = load_prediction_csv(str(rf_path), "y_pred")
    ar_df_sig = pd.DataFrame()  # aún no tienes ARIMA global

    # --- Inicializar SIEMPRE para evitar UnboundLocalError en debug ---
    dyn_pred_class = None
    dyn_conf = None
    pred_used = None

    pred_rf = None
    pred_rf_proba = None

    # --- Buscar predicción precomputada (CSV) ---
    if target_ts_g is not None and not rf_df_sig.empty and "datetime" in rf_df_sig.columns:
        row = rf_df_sig[rf_df_sig["datetime"] == target_ts_g]
        if row.empty:
            nearest = _nearest_row_by_ts(rf_df_sig, target_ts_g)
            if nearest is not None:
                pred_rf = nearest.get("y_pred", None)
                pred_rf_proba = nearest.get("proba_up", None)
        else:
            pred_rf = row["y_pred"].iloc[0] if "y_pred" in row.columns else None
            pred_rf_proba = row["proba_up"].iloc[0] if "proba_up" in row.columns else None

    # --- Predicción dinámica si el usuario presiona el botón ---
    dyn_pred_class = None
    dyn_conf = None
    if run_pred and target_ts_g is not None:
        target_ts_price = last_seen_ts + pd.Timedelta(hours=price_horizon_hours)
        with st.spinner("Ejecutando predicción dinámica (Fusion: sentimiento + precio)…"):
            fused = _run_fused_predict_cached(
                target_ts_sent_str=str(target_ts_g),
                target_ts_price_str=str(target_ts_price),
                resolution=interval,      # 1h/4h/24h (lo que elijas en granularidad)
                mode="auto",
                window_hours=24,
                sentiment_model_name="xgb_clf",   # deja que el backend elija defaults
                price_model_name="xgb_price",
            )

        # Guardar en session_state como antes
        st.session_state["__dyn_global_result__"] = fused.get("sentiment")
        st.session_state["__dyn_global_price_result__"] = fused.get("price")

        # Debug
        if debug_mode:
            with st.expander("Fusion result (debug)", expanded=False):
                st.write(fused)
    
    # ====================== BLOQUE FINAL GLOBAL (ORDEN: Sentimiento -> Precio -> Señal final) ======================
    # 1) Predicción técnica de PRECIO (ya la guardaste cuando run_pred)
    price_result = st.session_state.get("__dyn_global_price_result__")
    # Mostrar bloque de precio (t+interval o el horizon que uses)
    with col_prediction:
        render_price_block(price_result, horizon_label=interval, debug_mode=debug_mode)
    # 2) Construir señal final combinada (Precio + Sentimiento)
    price_cls = None
    price_conf = None
    if isinstance(price_result, dict):
        price_cls = price_result.get("prediction")
        price_conf = price_result.get("confidence")
    # sentiment_label ya existe en Global (Positivo/Negativo/Neutral) por tu cálculo con news
    final_label, final_color = combine_final_signal(price_cls, sentiment_label)
    # confianza final: prioriza precio si existe, si no usa la del modelo de sentimiento si existe, si no 50%
    final_conf = price_conf if price_conf is not None else dyn_conf
    try:
        prob_final = int(max(0, min(100, float(final_conf) * 100))) if final_conf is not None else 50
    except Exception:
        prob_final = 50
    # 3) Mostrar Señal final (COMBINADA)
    with col_final_signal:
        st.subheader("Señal final")
        st.markdown(f"<h3 style='color:{final_color}; margin:0'>{final_label}</h3>", unsafe_allow_html=True)
        st.progress(prob_final)

    # Debug opcional
    if debug_mode:
        with st.expander("Detalles técnicos (debug)", expanded=False):
            st.write({
                "sentiment_label": sentiment_label,
                "price_prediction": price_cls,
                "price_conf": price_conf,
                "dyn_sent_prediction": dyn_pred_class,
                "dyn_sent_conf": dyn_conf,
                "final_label": final_label
            })          

    st.divider()

    # ------------------ Tabs ------------------
    tab_prediction, tab_filtered_news, tab_details = st.tabs(["Predicción", "Noticias filtradas", "Detalles"])

    # --------- Tab Contexto ---------
    with tab_filtered_news:
        st.caption(f"Registros: {len(news_window)}")
        st.caption("💡 La señal resume el sentimiento de las noticias más recientes dentro del rango elegido.")
        if not news_window.empty:
            st.dataframe(news_window[["published_at", "source", "axis", "title"]].head(200), width='stretch')
            csv_bytes = news_window.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv_bytes, "news_filtered.csv", "text/csv")
        else:
            st.info("No hay noticias para mostrar en el rango.")
    # --------- Tab Señal & Predicción ---------
    with tab_prediction:
        st.markdown("Sentimiento:")
        st.caption("- Modelos de clasificación supervisada (Random Forest y XGBoost) entrenados sobre features de sentimiento de noticias + Fear & Greed, sin usar precios.")
        st.caption("- Predicen la dirección del movimiento (sube/baja) en t+1.")
        st.markdown("Precio:")
        st.caption("- Modelos de clasificación supervisada (Random Forest y XGBoost) entrenados sobre features técnicas derivadas solo de precios históricos (retornos, lags, volatilidad, momentum).")
        st.caption("- Predicen la dirección del movimiento (sube/baja) en t+1.")

        # --- Sentiment (precomputaded) ---
        rf_df  = load_pred_if_exists("rf_clf", interval, "y_pred")
        xgb_df = load_pred_if_exists("xgb_clf", interval, "y_pred")
        if not rf_df.empty:
            rf_df = rf_df.rename(columns={"y_pred": "y_pred_rf"})
        if not xgb_df.empty:
            xgb_df = xgb_df.rename(columns={"y_pred": "y_pred_xgb"})
        # --- Merge sentiment ---
        pred_sent_df = pd.DataFrame()
        if not rf_df.empty and not xgb_df.empty:
            pred_sent_df = pd.merge(rf_df, xgb_df, on="datetime", how="outer")
        else:
            pred_sent_df = rf_df if not rf_df.empty else xgb_df
        if not pred_sent_df.empty:
            pred_sent_df = pred_sent_df.sort_values("datetime")

        # --- Price (precomputaded) ---
        rfp_df  = load_pred_if_exists("rf_price", interval, "y_pred")
        xgbp_df = load_pred_if_exists("xgb_price", interval, "y_pred")

        if not rfp_df.empty:
            rfp_df = rfp_df.rename(columns={"y_pred": "y_pred_rf_price"})
        if not xgbp_df.empty:
            xgbp_df = xgbp_df.rename(columns={"y_pred": "y_pred_xgb_price"})
        # --- Merge price ---
        pred_price_df = pd.DataFrame()
        if not rfp_df.empty and not xgbp_df.empty:
            pred_price_df = pd.merge(rfp_df, xgbp_df, on="datetime", how="outer")
        else:
            pred_price_df = rfp_df if not rfp_df.empty else xgbp_df
        if not pred_price_df.empty:
            pred_price_df = pred_price_df.sort_values("datetime")
    
        if rf_df.empty and xgb_df.empty:
            st.info("Aún no hay CSVs de predicción de sentimiento en data/processed/.")
        if rfp_df.empty and xgbp_df.empty:
            st.info(f"No hay CSVs de predicción de PRECIO para {interval} en data/processed/.")
        else:
            # Merge sentiment
            pred_sent_df = pd.DataFrame()
            if not rf_df.empty and not xgb_df.empty:
                pred_sent_df = pd.merge(rf_df, xgb_df, on="datetime", how="outer")
            else:
                pred_sent_df = rf_df if not rf_df.empty else xgb_df

            if not pred_sent_df.empty:
                pred_sent_df = pred_sent_df.sort_values("datetime")
            # Merge price
            pred_price_df = pd.DataFrame()
            if not rfp_df.empty and not xgbp_df.empty:
                pred_price_df = pd.merge(rfp_df, xgbp_df, on="datetime", how="outer")
            else:
                pred_price_df = rfp_df if not rfp_df.empty else xgbp_df

            if not pred_price_df.empty:
                pred_price_df = pred_price_df.sort_values("datetime")

            # Ventana sentimiento (solo si existe datetime)
            if (pred_sent_df is None) or pred_sent_df.empty or ("datetime" not in pred_sent_df.columns):
                sent_pred_window = pd.DataFrame()
            else:
                sent_pred_window = pred_sent_df[(pred_sent_df["datetime"] >= start_ts) & (pred_sent_df["datetime"] <= end_ts)].copy()

                # Heatmap
                st.markdown("### Predicciones de sentimientos (visualización)")
                heat_df = sent_pred_window.copy()
                heat_df = heat_df[["datetime", "y_pred_rf", "y_pred_xgb"]].melt(
                    id_vars="datetime",
                    var_name="model",
                    value_name="pred"
                ).dropna()
                heat = alt.Chart(heat_df).mark_rect().encode(
                    x=alt.X("yearmonthdate(datetime):T", title="Día"),
                    y=alt.Y("model:N", title="Sentiment"),
                    color=alt.Color(
                        "mean(pred):Q",
                        scale=alt.Scale(domain=[0, 1], range=["#b91e0d", "#22a884"]),
                        legend=alt.Legend(title="Pred (promedio)")
                    ),
                    tooltip=[
                        alt.Tooltip("yearmonthdate(datetime):T", title="Día"),
                        alt.Tooltip("model:N", title="Sentiment"),
                        alt.Tooltip("mean(pred):Q", title="Promedio", format=".2f")
                    ]
                ).properties(height=120)
                st.altair_chart(heat, width="stretch")

                # Table
                st.markdown("### Predicciones de sentimientos (datos tabulados)")
                st.dataframe(sent_pred_window.head(200), width='stretch')
                csv_bytes = sent_pred_window.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar predicciones de sentimiento (CSV)", csv_bytes, "predictions_filtered.csv", "text/csv", key="download_sentiment")

            # Ventana precios (solo si existe datetime)
            if (pred_price_df is None) or pred_price_df.empty or ("datetime" not in pred_price_df.columns):
                price_pred_window = pd.DataFrame()
            else:
                price_pred_window = pred_price_df[(pred_price_df["datetime"] >= start_ts) & (pred_price_df["datetime"] <= end_ts)].copy()

                # Heatmap
                st.markdown("### Predicciones de precios (visualización)")
                heat_df = price_pred_window.copy()
                heat_df = heat_df[["datetime", "y_pred_rf_price", "y_pred_xgb_price"]].melt(
                    id_vars="datetime",
                    var_name="model",
                    value_name="pred"
                ).dropna()
                heat = alt.Chart(heat_df).mark_rect().encode(
                    x=alt.X("yearmonthdate(datetime):T", title="Día"),
                    y=alt.Y("model:N", title="Price"),
                    color=alt.Color(
                        "mean(pred):Q",
                        scale=alt.Scale(domain=[0, 1], range=["#b91e0d", "#22a884"]),
                        legend=alt.Legend(title="Pred (promedio)")
                    ),
                    tooltip=[
                        alt.Tooltip("yearmonthdate(datetime):T", title="Día"),
                        alt.Tooltip("model:N", title="Price"),
                        alt.Tooltip("mean(pred):Q", title="Promedio", format=".2f")
                    ]
                ).properties(height=120)
                st.altair_chart(heat, width="stretch")

                # Table
                st.markdown("### Predicciones de precios (datos tabulados)")
                st.dataframe(price_pred_window.head(200), width='stretch')
                csv_bytes = price_pred_window.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar predicciones de precios (CSV)", csv_bytes, "predictions_filtered.csv", "text/csv", key="download_price")

    # --------- Tab Detalles ---------
    with tab_details:
        st.subheader("Resumen de datos")
        fuente = "raw" if "raw" in str(price_path).lower() else "processed"
        rows_news = int(len(news_window))
        sources = int(news_window["source"].nunique()) if (not news_window.empty and "source" in news_window.columns) else 0

        st.write({
            "fuente": fuente,
            "rows_prices": int(len(price_window)),
            "rows_news": rows_news,
            "sources_unicas": sources
        })

# ------------------ Top nav (streamlit-option-menu) ------------------
manual = st.session_state.get("nav_manual_select", None)
selected = option_menu(
    menu_title=None,
    options=["Home", "Dashboard", "Overview"],
    icons=["house", "graph-up", "card-text", "lightbulb"],
    default_index=0 if manual is None else manual,
    orientation="horizontal",
    manual_select=manual,
    styles={
        "container": {"padding": "0.2rem 0.2rem", "background-color": "#0b1120"},
        "icon": {"color": "#e5e7eb", "font-size": "16px"},
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px 6px",
                     "color": "#94a3b8", "border-radius": "10px", "padding": "6px 12px"},
        "nav-link-selected": {"background-color": "#111827", "color": "#e5e7eb"}
    },
)

if selected in ("Home", "Overview"):
    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)

if manual is not None:
    st.session_state["nav_manual_select"] = None

if selected == "Home":
    render_home()
elif selected == "Overview":
    render_overview()
elif selected == "Dashboard":
    render_dashboard()

