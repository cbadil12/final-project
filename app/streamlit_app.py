# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone, time as dtime
import sys
import os
from pathlib import Path
import base64

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import math

# ------------------ Paths base ------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ROOT_PATH = Path(ROOT)
ASSETS_DIR = ROOT_PATH / "assets"
RAW_DIR = ROOT_PATH / "data" / "raw"
PROCESSED_DIR = ROOT_PATH / "data" / "processed"

# --- Import dinámico ---
DYN_IMPORT_ERROR = None
run_dynamic_predict = None
run_fused_predict = None

try:
    from src.backend.dynamic_predict import run_dynamic_predict
except Exception as e:
    run_dynamic_predict = None
    DYN_IMPORT_ERROR = f"import run_dynamic_predict failed: {repr(e)}"

# Import opcional: si no existe, NO rompe el resto
try:
    from src.backend.dynamic_predict import run_fused_predict
except Exception as e:
    run_fused_predict = None
    if DYN_IMPORT_ERROR:
        DYN_IMPORT_ERROR += f" | import run_fused_predict failed: {repr(e)}"
    else:
        DYN_IMPORT_ERROR = f"import run_fused_predict failed: {repr(e)}"

# Helpers de frontend
from src.frontend.data_loader import (
    load_prices, build_ohlc, load_news, filter_news_timewindow,
    default_relevance_filter, quick_sentiment_score, load_prediction_csv
)
from src.frontend.ui_components import candlestick_chart, line_with_ma

# ------------------ Files esperados ------------------
# raw (fuentes base)
PRICE_1H_RAW_PATH = RAW_DIR / "btcusd-1h.csv"
PRICE_4H_RAW_PATH = RAW_DIR / "btcusd-4h.csv"
NEWS_RAW_PATH     = RAW_DIR / "news_raw.csv"
FNG_RAW_PATH      = RAW_DIR / "fear_greed.csv"

# processed (datasets con features + targets para modelos sentimiento)
DATASET_SENT_1H_PATH = PROCESSED_DIR / "dataset_sentiment_target_1h.csv"
DATASET_SENT_4H_PATH = PROCESSED_DIR / "dataset_sentiment_target_4h.csv"

# predictions (outputs sentimiento)
PRED_RF_1H_PATH  = PROCESSED_DIR / "predictions_rf_clf_1h.csv"
PRED_RF_4H_PATH  = PROCESSED_DIR / "predictions_rf_clf_4h.csv"
PRED_XGB_1H_PATH = PROCESSED_DIR / "predictions_xgb_clf_1h.csv"
PRED_XGB_4H_PATH = PROCESSED_DIR / "predictions_xgb_clf_4h.csv"

# (opcional) dataset “solo sentimiento”
SENTIMENT_ONLY_1H_PATH = PROCESSED_DIR / "sentiment_only_1h.csv"

def pred_path(model_name: str, resolution: str) -> Path:
    """
    Devuelve el path del CSV de predicciones precomputadas si existe.
    Formato esperado: data/processed/predictions_{model_name}_{resolution}.csv
    """
    p = PROCESSED_DIR / f"predictions_{model_name}_{resolution}.csv"
    return p

def load_pred_if_exists(model_name: str, resolution: str, y_col: str = "y_pred") -> pd.DataFrame:
    p = pred_path(model_name, resolution)
    if not p.exists():
        return pd.DataFrame()

    df = load_prediction_csv(str(p), y_col)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normaliza timestamp como columna
    if "timestamp" not in df.columns:
        # Caso: timestamp viene como índice
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "timestamp"})
        # Caso alternativo: columna datetime
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})

    # Fuerza tipo datetime UTC si existe
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    return df

# ------------------ Funciones de utilidad ------------------
def _asset_to_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@st.cache_data(show_spinner=False)
def _load_news_cached(path_str: str) -> pd.DataFrame:
    return load_news(path_str)

@st.cache_data(show_spinner=False)
def _load_prices_compat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Caso A: btcusd-1h.csv (datetime, open, high, low, close, volume)
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df["price"] = pd.to_numeric(df.get("close"), errors="coerce")
        return df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    # Caso B: btcusd-4h.csv separado por ;
    if df.shape[1] == 1 and isinstance(df.iloc[0,0], str) and ";" in df.iloc[0,0]:
        df = pd.read_csv(path, sep=";", header=None)
        df.columns = ["date", "time", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True, utc=True, errors="coerce")
        df["price"] = pd.to_numeric(df["close"], errors="coerce")
        return df.dropna(subset=["timestamp"]).sort_values("timestamp")
    
    # Caso C: fallback a loader original
    return load_prices(path)


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

# ------------------ Config Streamlit ------------------
st.set_page_config(page_title='BTC Predictor', page_icon='📈', layout='wide')

# ----- Bootstrap Icons -----
st.markdown(
    """
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    """,
    unsafe_allow_html=True
)

# --- Estilos globales + Home hero ---
btc_bg_uri = _asset_to_data_uri(ASSETS_DIR / "btc_logo_gray.png")

st.markdown(
    f"""
    <style>
    .main {{ background-color: #0F1015; color: #e5e7eb; }}
    .stMetric {{ background-color: #0F1015 !important; }}
    
    /* Portada ("Home") */
    .hero {{
      min-height: 72vh; padding: 4rem 2rem; border-radius: 16px;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background-color: #0F1015; color: #e5e7eb; position: relative; overflow: hidden;
    }}
    .hero::before{{
      content:""; position:absolute; inset:0; background-repeat: repeat;
      background-size: 220px; background-position: 0 0; opacity: 0.08;
      filter: grayscale(100%); background-image: url('{btc_bg_uri if btc_bg_uri else ""}');
    }}
    .hero h1{{ margin:0 0 .5rem 0; font-size: clamp(2.2rem, 4vw, 3.4rem); z-index:1; }}
    .hero .tagline{{ color:#94a3b8; font-size: 1.05rem; text-align:center; max-width: 900px; z-index:1; }}
    .hero .cta{{ margin-top: 1.5rem; z-index:1; }}

    /* Disclaimer message (footer global) */
    .disclaimer {{
    margin-top: 0.75rem;
    color: #94a3b8;
    font-size: 0.78rem;
    opacity: 0.90;
    text-align: center;
    line-height: 1.25;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] > div {{
        display: flex; flex-direction: column; height: 100vh;
    }}
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {{
        display: flex; flex-direction: column; height: 100%;
    }}
    .sidebar-footer {{
        margin-top: auto; padding-top: 12px; border-top: 1px solid #2a2f3a;
        color: #94a3b8; font-size: 0.78rem; opacity: 0.90; line-height: 1.25;
    }}

    .sidebar-footer .meta {{ margin-bottom: 6px; opacity: 0.95; }}
    .sidebar-footer .disclaimer {{
        margin-top: 6px; font-size: 0.74rem; opacity: 0.85; text-align: left;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

NAV_INDEX = {"Home": 0, "Overview": 1, "Dashboard": 2, "Recomendaciones": 3}
def go_to(page_name: str):
    st.session_state["nav_manual_select"] = NAV_INDEX.get(page_name, 0)

DISCLAIMER_TEXT = (
    "**Aviso: BTC Predictor es un proyecto demostrativo/educativo. "
    "La información mostrada es solo informativa y no constituye asesoramiento financiero.**"
)
def render_disclaimer():
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)

# ------------------ Definición de regímenes ------------------
REGIME_BOUNDS = [
    ("2012–2016", pd.Timestamp("2012-11-28 00:00:00", tz="UTC"), pd.Timestamp("2016-07-09 23:59:59", tz="UTC")),
    ("2016–2020", pd.Timestamp("2016-07-10 00:00:00", tz="UTC"), pd.Timestamp("2020-05-10 23:59:59", tz="UTC")),
    ("2020–2024", pd.Timestamp("2020-05-11 00:00:00", tz="UTC"), pd.Timestamp("2024-04-19 23:59:59", tz="UTC")),
    ("2024–Actual", pd.Timestamp("2024-04-20 00:00:00", tz="UTC"), pd.Timestamp("2100-12-31 23:59:59", tz="UTC")),
]

def regime_for_timestamp(ts: pd.Timestamp) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    for name, r_start, r_end in REGIME_BOUNDS:
        if r_start <= ts <= r_end:
            return name
    return None

def bounds_for_regime(name: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    for r_name, r_start, r_end in REGIME_BOUNDS:
        if r_name == name:
            return r_start, r_end
    return None


def round_down_to_interval(ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    if interval == "1h":
        return ts.replace(minute=0, second=0, microsecond=0)
    if interval == "4h":
        hr = (ts.hour // 4) * 4
        return ts.replace(hour=hr, minute=0, second=0, microsecond=0)
    if interval == "24h":
        # borde inferior del día UTC
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    return ts



def _nearest_row_by_ts(df: pd.DataFrame, target_ts: pd.Timestamp, ts_col="timestamp"):
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

def _get_base_price_for_target(price_pack, fallback_df, target_ts: pd.Timestamp, horizon_hours: int):
    """Intenta base_ref = close en (target_ts - horizon). Si no existe, usa último close <= target_ts."""
    # Preferimos OHLC si existe
    ohlc = getattr(price_pack, "ohlc", None)
    if ohlc is not None and not ohlc.empty and "timestamp" in ohlc.columns:
        ohlc = ohlc.sort_values("timestamp")
        t_prev = target_ts - pd.Timedelta(hours=horizon_hours)
        exact = ohlc[ohlc["timestamp"] == t_prev]
        if not exact.empty and "close" in exact.columns:
            return float(exact["close"].iloc[0])

        prev = ohlc[ohlc["timestamp"] <= target_ts].tail(1)
        if not prev.empty and "close" in prev.columns:
            return float(prev["close"].iloc[0])

    # Fallback: usar price_window/df original
    if fallback_df is not None and not fallback_df.empty and "timestamp" in fallback_df.columns and "price" in fallback_df.columns:
        tmp = fallback_df.dropna(subset=["price"]).sort_values("timestamp")
        prev = tmp[tmp["timestamp"] <= target_ts].tail(1)
        if not prev.empty:
            return float(prev["price"].iloc[0])

    return None


def _sentiment_label_to_class(sentiment_label: str | None):
    """Mapea etiqueta de sentimiento a clase: Positivo=1, Negativo=0, Neutral=None."""
    if sentiment_label == "Positivo":
        return 1
    if sentiment_label == "Negativo":
        return 0
    return None  # Neutral o desconocido

def combine_final_signal(price_pred: int | None, sentiment_label: str | None):
    """
    Regla final (la que definiste):
    - Precio SUBIDA + Sentimiento Positivo/Neutral => SUBIDA
    - Precio BAJADA + Sentimiento Negativo/Neutral => BAJADA
    - Si chocan => ESCENARIO MIXTO
    - Si falta uno => usa el que esté disponible, si no => NEUTRAL
    """
    sent_cls = _sentiment_label_to_class(sentiment_label)

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

# ------------------ Vistas ------------------
def render_home():
    st.markdown(
       """
        <div class="hero">
          <h1>BTC Predictor</h1>
          <p class="tagline">
            Predicción técnica + sentimiento para una señal de mercado más completa
          </p>
          <div class="cta">
       """,
        unsafe_allow_html=True
    )
    render_disclaimer()

OVERVIEW_MD = """
## BTC Predictor

En un mercado altamente volátil como el de Bitcoin, las variaciones de precio suelen 
estar impulsadas no solo por factores técnicos, sino también por el contexto informativo 
y emocional del mercado.  
**BTC Predictor** combina **ambos mundos** para ofrecer una visión más completa del comportamiento del precio.

---

### 🔎 ¿Qué integra?

- **Datos cuantitativos:** precio, estructura OHLC, series temporales y estacionalidad por ciclos (halvings).  
- **Datos cualitativos:** noticias filtradas y un análisis de sentimiento basado en ventanas recientes.  

Esta combinación permite observar **qué está haciendo el precio** y **qué lo puede estar impulsando**.

---

### 🎯 ¿Qué hace el predictor?

BTC Predictor permite dos enfoques de análisis complementarios:

#### **1. Modo Global**
Analiza un rango seleccionado por el usuario y construye:

- Precio histórico (1h / 4h) con resampling OHLC  
- Noticias filtradas por relevancia  
- Sentimiento agregado en ventanas recientes  
- **Predicción técnica t+1** (Random Forest y ARIMA global)

*Ideal para analizar periodos amplios y observar cómo cambian las señales en función del sentimiento y la acción del precio.*



#### **2. Modo Regímenes (Halvings)**
El usuario selecciona una fecha objetivo y la aplicación identifica automáticamente 
el **régimen al que pertenece**, aplicando el **modelo ARIMA específico de ese ciclo**.

Permite:
- evaluar la estacionalidad entre halvings,  
- comparar predicción vs precio real en periodos pasados,  
- y estudiar cómo cambian los patrones entre ciclos.

---

### 📌 Enfoque del proyecto

El objetivo es mostrar un **modelo predictivo funcional**, construido con técnicas 
de series temporales y enriquecido con información contextual.  
La aplicación facilita la exploración visual y comparativa entre modos, 
permitiendo entender cómo reaccionan los modelos ante diferentes condiciones del mercado.

Este diseño permite extender el sistema fácilmente con nuevos modelos, fuentes de datos
o flujos automatizados en versiones futuras.
"""

def render_overview():
    st.markdown(OVERVIEW_MD)
    left, center, right =st.columns(3)
    center.button("Ir a la App →", type="primary", on_click=go_to, args=("Dashboard",),)
    st.markdown("</div></div>", unsafe_allow_html=True)
    render_disclaimer()


def render_recommendations():
    st.header("🔮 Recomendaciones y mejoras futuras")

    st.markdown("""
    Esta sección resume **extensiones naturales** de la solución, centradas en 
    refinar y escalar lo que ya funciona en **BTC Predictor**, a futuro.

    ---

    ### 📡 1. Integración con datos en tiempo real
    - Conectar una API de precios en vivo. Limitaciones actuales por rango.
    - Añadir el indicador *Fear & Greed Index* desde su API oficial.
    - Actualizar las noticias automáticamente para que la señal de sentimiento
      se alimente del mercado actual, de una manera más rápida.

    

    ### 🔄 2. Automatizar el pipeline de predicción
    - Generar predicciones bajo demanda para fechas específicas.
    - Registrar entradas/salidas para evaluar históricamente el rendimiento del predictor.

    

    ### 📏 3. Evaluación y calibración continua
    - Implementar backtesting para comparar las predicciones con el precio real.
    - Analizar el rendimiento de cada régimen

    

    ### 💬 4. Extender el análisis de sentimiento
    - Incluir nuevas fuentes como redes sociales o el índice *Fear & Greed* diario.
    - Añadir ventanas ajustables por tipo de evento (últimas 4h, 24h, etc.)
    - Incorporar mayor granularidad en el peso de cada noticia, y mejorar el componente cualitativo sin cambiar el modelo predictivo.

    

    ### 🎛️ 5. Mejoras de experiencia de usuario
    - Agregar un modo “principiante” con explicaciones guiadas.
    - Resaltado visual cuando los modelos coinciden o divergen.
    - Exportación de vistas en PDF.

    ---

    En conjunto, estas recomendaciones **no cambian la lógica del predictor actual**, 
    sino que lo fortalecen y lo preparan para escenarios más dinámicos y escalables.
    """)
    render_disclaimer()


def render_app():
    st.title('BTC Predictor')
    st.caption("Dashboard para analizar precio, noticias y sentimiento del mercado en Bitcoin ₿.")

    # Abierto (una vez por sesión)
    if "opened_at_utc" not in st.session_state:
        st.session_state["opened_at_utc"] = datetime.now(timezone.utc)

    # -------------- Sidebar --------------
    st.sidebar.title("Panel de control")

    debug_mode = st.sidebar.toggle("Mostrar detalles técnicos (debug)", value=False)

    if debug_mode and DYN_IMPORT_ERROR:
        st.sidebar.error(f"Error import dynamic_predict: {DYN_IMPORT_ERROR}")
        st.sidebar.write({
            "run_dynamic_predict_is_none": run_dynamic_predict is None,
            "DYN_IMPORT_ERROR": DYN_IMPORT_ERROR
        })


    # Modo (Global vs Regímenes) — Global por defecto
    mode = st.sidebar.selectbox(
        "Modo de análisis",
        ["Global", "Regímenes (Halvings)"],
        index=0,
        key="mode",
    )

    # Granularidad
    interval_options = ["1h", "4h", "24h"]
    interval = st.sidebar.selectbox(
        "Granularidad (gráfico)",
        interval_options,
        index=interval_options.index(st.session_state.get("interval", "1h"))
            if st.session_state.get("interval", "1h") in interval_options else 0,
        key="interval",
        help="Frecuencia del precio para el gráfico (OHLC/linea)."
    )

    INTERVAL_TO_HOURS = {"1h": 1, "4h": 4, "24h": 24}
    horizon_hours = INTERVAL_TO_HOURS.get(interval, 1)
    st.sidebar.caption(f"Predicción sentimiento: **t+1** (siguiente {interval} → +{horizon_hours}h)")

    
    # Horizonte precio = igual que granularidad del gráfico (simple y coherente)
    price_horizon = interval
    price_horizon_hours = horizon_hours

    
    # PENDIENTE si se borra o no ---------------- Horizonte (para el MODELO DE PRECIO) ----------------
    #price_horizon_options = ["1h", "4h", "24h"]
    #price_horizon = st.sidebar.selectbox(
        #"Horizonte predicción (precio)",
        #price_horizon_options,
        #index=price_horizon_options.index(st.session_state.get("price_horizon", "24h"))
            #if st.session_state.get("price_horizon", "24h") in price_horizon_options else 2,
        #key="price_horizon",
        #help="Horizonte del modelo de precio (dirección). 24h funciona como proxy 'diario' más estable."
    #)

    #PRICE_HORIZON_TO_HOURS = {"1h": 1, "4h": 4, "24h": 24}
    #price_horizon_hours = PRICE_HORIZON_TO_HOURS.get(price_horizon, 24)
    #st.sidebar.caption(f"Predicción precio: **t+{price_horizon}** (→ +{price_horizon_hours}h)")

    # -------------- Nueva selección de datasets según granularidad --------------
    price_path = PRICE_1H_RAW_PATH
    
    news_path = NEWS_RAW_PATH

    # -------------- Carga de datasets (con caché) --------------
    prices_df = _load_prices_compat(str(price_path))
    news_df   = _load_news_cached(str(news_path)) if news_path.exists() else pd.DataFrame()

    if prices_df.empty:
        st.error("No se pudieron cargar precios (revisa data/raw y/o data/processed).")
        return

    # -------------- Widgets dependientes del modo --------------
    # Global: rango
    if mode == "Global":
        # Rango por defecto (60 días hasta el max de datasets)
        if not news_df.empty:
            min_ts = min(prices_df["timestamp"].min(), news_df["published_at"].min())
            max_ts = max(prices_df["timestamp"].max(), news_df["published_at"].max())
        else:
            min_ts = prices_df["timestamp"].min()
            max_ts = prices_df["timestamp"].max()

        default_end = max_ts.date()
        default_start = (max_ts - pd.Timedelta(days=60)).date()

        start_date = st.sidebar.date_input(
            "Fecha inicio", value=st.session_state.get("start_date", default_start),
            key="start_date"
        )
        end_date = st.sidebar.date_input(
            "Fecha fin", value=st.session_state.get("end_date", default_end),
            key="end_date"
        )
        relevance = st.sidebar.checkbox(
            "Filtrar noticias relevantes",
            value=st.session_state.get("relevance", True),
            key="relevance",
            help="Aplica filtro simple ‘bitcoin|btc’ en título/descripcion."
        )

        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        end_ts   = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Filtros data
        price_window = prices_df[(prices_df["timestamp"] >= start_ts) & (prices_df["timestamp"] <= end_ts)].copy()

        if not news_df.empty:
            news_window = filter_news_timewindow(news_df, start_ts, end_ts)
            if relevance:
                news_window = default_relevance_filter(news_window)
        else:
            news_window = pd.DataFrame()

        # Sidebar footer
        opened = st.session_state["opened_at_utc"]
        st.sidebar.markdown(
            f"""
            <div class="sidebar-footer">
                <div class="meta"><b>Abierto:</b> {opened:%Y-%m-%d %H:%M} UTC</div>
                <div class="meta"><b>Datos hasta:</b> {prices_df['timestamp'].max():%Y-%m-%d %H:%M} UTC</div>
                <div class="meta"><p></p></div>
                <div class="disclaimer">{DISCLAIMER_TEXT}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------*----- Regímenes: fecha objetivo (y hora) -----*----------
    else:
        # Elegir fecha/ hora objetivo (UTC) — front asigna régimen automáticamente
        now_utc = pd.Timestamp.utcnow()

        tgt_date = st.sidebar.date_input(
            "Fecha objetivo (UTC)",
            value=now_utc.date(),
            key="target_date"
        )
        default_time = dtime(hour=now_utc.hour, minute=0, second=0)
        tgt_time = st.sidebar.time_input(
            "Hora objetivo (UTC)",
            value=default_time,
            key="target_time"
        )
        target_naive = datetime.combine(pd.to_datetime(tgt_date).date(), tgt_time)
        target_ts = pd.Timestamp(target_naive).tz_localize("UTC")
        target_ts = round_down_to_interval(target_ts, interval)

        # Determinar régimen por target_ts
        regime = regime_for_timestamp(target_ts) or "2024–Actual"
        r_bounds = bounds_for_regime(regime)
        if r_bounds:
            r_start, r_end = r_bounds
            # Clamp a límites del régimen + realinear
            if target_ts < r_start: target_ts = round_down_to_interval(r_start, interval)
            if target_ts > r_end:   target_ts = round_down_to_interval(r_end, interval)
        else:
            r_start, r_end = pd.Timestamp("2012-01-01", tz="UTC"), pd.Timestamp("2100-12-31", tz="UTC")

        st.sidebar.caption(f"Régimen detectado: **{regime}**")

        # Para el gráfico de contexto, mostramos el tramo del régimen
        price_window = prices_df[(prices_df["timestamp"] >= r_start) & (prices_df["timestamp"] <= min(prices_df['timestamp'].max(), r_end))].copy()
        news_window = pd.DataFrame()  # no aplica a pred futura por diseño

        # Guardar en session para usar en tabs
        st.session_state["__regime_target_ts__"] = target_ts
        st.session_state["__regime_name__"]      = regime
        st.session_state["__regime_bounds__"]    = (r_start, r_end)

        # Sidebar footer
        opened = st.session_state["opened_at_utc"]
        st.sidebar.markdown(
            f"""
            <div class="sidebar-footer">
                <div class="meta"><b>Abierto:</b> {opened:%Y-%m-%d %H:%M} UTC</div>
                <div class="meta"><b>Régimen:</b> {regime} ({r_start:%Y-%m-%d} → {r_end:%Y-%m-%d})</div>
                <div class="meta"><p></p></div>
                <div class="disclaimer">{DISCLAIMER_TEXT}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown("---")
    run_pred = st.sidebar.button(
        "🚀 Ejecutar Predicción",
        help="Ejecuta la predicción live usando los modelos disponibles."
    )
    # --- Feedback inmediato del botón ---
    if run_pred:
        st.toast("Ejecutando predicción dinámica…", icon="🚀")


    st.sidebar.markdown("---")
    if st.sidebar.button(
        "Actualizar datos",
        type="secondary",
        help="Refresca la información si se actualizaron CSVs en data/processed (limpia caché y recarga)."
    ):
        st.cache_data.clear()
        st.rerun()

    # ------------------ Precio (OHLC o línea) ------------------
    price_pack = build_ohlc(price_window, interval=interval, min_points_per_candle=2)

    col_price, col_stats = st.columns([2, 1])

    with col_price:
        st.subheader("Precio BTC")
        if price_pack.mode in ("ohlc", "computed_ohlc") and not price_pack.ohlc.empty:
            chart = candlestick_chart(price_pack.ohlc, ts_col="timestamp").properties(height=380)
            st.altair_chart(chart, width='stretch')
        else:
            non_null = price_window.dropna(subset=["price"]).copy()
            if not non_null.empty:
                chart = line_with_ma(non_null, ts_col="timestamp", price_col="price", ma_window=24).properties(height=380)
                st.altair_chart(chart, width='stretch')
            else:
                st.warning("No hay datos de precio disponibles en el rango/regimen mostrado.")

    # ------------------ Panel Sentimiento / Señal ------------------
    with col_stats:
        st.subheader("Sentimiento del mercado")

        if mode == "Global":
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
            st.metric("Sentimiento agregado", sentiment_label, f"{sentiment_score:+.2f}")
            st.progress(prob)
            st.caption(f"Probabilidad (escala interna): {prob}%")
            st.caption(f"Hits → Positivos: {pos_hits} | Negativos: {neg_hits}")
            st.caption(f"Ventana usada para el sentimiento: {used_scope}")

            # --- Señal final (Global) basada en predicción t+1 ---
            last_series = price_window.dropna(subset=["price"])
            last_price = float(last_series.iloc[-1]["price"]) if not last_series.empty else None
            last_seen_ts = last_series["timestamp"].max() if not last_series.empty else None
            
            target_ts_g = None
            if last_seen_ts is not None:
                target_ts_g = last_seen_ts + pd.Timedelta(hours=horizon_hours)


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
            
            if target_ts_g is not None and not rf_df_sig.empty and "timestamp" in rf_df_sig.columns:
                row = rf_df_sig[rf_df_sig["timestamp"] == target_ts_g]
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
            render_price_block(price_result, horizon_label=price_horizon, debug_mode=debug_mode)

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
            st.subheader("Señal final")
            st.markdown(
                f"<h3 style='color:{final_color}; margin:0'>{final_label}</h3>",
                unsafe_allow_html=True
            )
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
 
        else:
            # Regímenes: usar predicción dinámica al pulsar botón; fallback a tendencia si no hay resultado
            target_ts = st.session_state.get("__regime_target_ts__")
            regime    = st.session_state.get("__regime_name__")
            st.info("En modo Regímenes, la predicción dinámica decide si usa histórico o live según disponibilidad de datos.")

            final_label, color = "ESCENARIO NEUTRAL", "gray"
            conf = None

            # # Precio base: último observado del tramo mostrado (para fallback)
            last_series = price_window.dropna(subset=["price"]).sort_values("timestamp")
            base_ref = float(last_series.iloc[-1]["price"]) if not last_series.empty else None

            # --- Ejecutar predicción dinámica SOLO si el usuario presiona el botón ---
            dyn_result = st.session_state.get("__dyn_reg_result__")
            price_result_reg = st.session_state.get("__dyn_reg_price_result__")

            if run_pred and target_ts is not None:
                target_ts_price = target_ts + pd.Timedelta(hours=price_horizon_hours)

                with st.spinner("Ejecutando predicción dinámica (Fusion: sentimiento + precio)…"):
                    fused = _run_fused_predict_cached(
                        target_ts_sent_str=str(target_ts),
                        target_ts_price_str=str(target_ts_price),
                        resolution=interval,      # 1h/4h/24h (lo que elijas en granularidad)
                        mode="auto",
                        window_hours=24,
                        sentiment_model_name="xgb_clf",   # deja que el backend elija defaults
                        price_model_name="xgb_price",
                    )

                # Guardar en session_state como antes
                st.session_state["__dyn_reg_result__"] = fused.get("sentiment")
                st.session_state["__dyn_reg_price_result__"] = fused.get("price")

                # Feedback opcional
                if debug_mode:
                    with st.expander("Fusion result (debug)", expanded=False):
                        st.write(fused)

            # ====================== BLOQUE FINAL REGÍMENES (ORDEN: Sentimiento -> Precio -> Señal final) ======================

            # 1) Derivar etiqueta de sentimiento desde dyn_result (aquí no usamos news_window)
            sentiment_label_reg = "Neutral"
            sent_conf = None

            if isinstance(dyn_result, dict):
                pred_class = dyn_result.get("prediction", None)
                sent_conf = dyn_result.get("confidence", None)
                if pred_class == 1:
                    sentiment_label_reg = "Positivo"
                elif pred_class == 0:
                    sentiment_label_reg = "Negativo"

            # Mostrar el bloque de "Predicción técnica (precio)" usando el helper que ya tienes
            price_result_reg = st.session_state.get("__dyn_reg_price_result__")
            render_price_block(price_result_reg, horizon_label=price_horizon, debug_mode=debug_mode)

            # 2) Construir señal final combinada (Precio + Sentimiento)
            price_cls = None
            price_conf = None
            if isinstance(price_result_reg, dict):
                price_cls = price_result_reg.get("prediction", None)
                price_conf = price_result_reg.get("confidence", None)

            final_label, final_color = combine_final_signal(price_cls, sentiment_label_reg)

            # 3) Confianza final: prioriza precio si existe, si no sentimiento, si no 50%
            final_conf = price_conf if price_conf is not None else sent_conf
            try:
                prog = int(max(0, min(100, float(final_conf) * 100))) if final_conf is not None else 50
            except Exception:
                prog = 50

            # 4) Mostrar Señal final (COMBINADA)
            st.subheader("Señal final")
            st.markdown(
                f"<h3 style='color:{final_color}; margin:0'>{final_label}</h3>",
                unsafe_allow_html=True
            )
            st.progress(prog)

            # Debug opcional
            if debug_mode:
                with st.expander("Detalles técnicos (debug)", expanded=False):
                    st.write({
                        "regime": regime,
                        "target_ts": str(target_ts),
                        "sentiment_label_reg": sentiment_label_reg,
                        "price_prediction": price_cls,
                        "final_label": final_label,
                        "dyn_sent_result": dyn_result,
                        "dyn_price_result": price_result_reg,
                    })          

    st.divider()

    # ------------------ Tabs ------------------
    tab_signal, tab_context, tab_method, tab_details = st.tabs(["Señal & Predicción", "Contexto", "Metodología", "Detalles"])

    # --------- Tab Contexto ---------
    with tab_context:
        st.subheader("Noticias filtradas")
        if mode == "Global":
            st.caption(f"Registros: {len(news_window)}")
            st.caption("💡 La señal resume el sentimiento de las noticias más recientes dentro del rango elegido.")
            if not news_window.empty:
                st.dataframe(news_window[["published_at", "source", "axis", "title"]].head(200), width='stretch')
                csv_bytes = news_window.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar CSV", csv_bytes, "news_filtered.csv", "text/csv")
            else:
                st.info("No hay noticias para mostrar en el rango.")
        else:
            st.info("En modo Regímenes, el contexto de noticias no se muestra aquí (se usa predicción dinámica bajo demanda).")

    # --------- Tab Señal & Predicción ---------
    with tab_signal:
        st.subheader("Predicción")
        st.caption("Sentimiento y Precio: RF/XGB (clasificación)")

        # Carga flexible según granularidad
        # --- Sentiment (precomputado) ---
        rf_df  = load_pred_if_exists("rf_clf", interval, "y_pred")
        xgb_df = load_pred_if_exists("xgb_clf", interval, "y_pred")

        if not rf_df.empty:
            rf_df = rf_df.rename(columns={"y_pred": "y_pred_rf"})
        if not xgb_df.empty:
            xgb_df = xgb_df.rename(columns={"y_pred": "y_pred_xgb"})

        # --- Price (precomputado) ---
        rfp_df  = load_pred_if_exists("rf_price", interval, "y_pred")
        xgbp_df = load_pred_if_exists("xgb_price", interval, "y_pred")

        if not rfp_df.empty:
            rfp_df = rfp_df.rename(columns={"y_pred": "y_pred_rf_price"})
        if not xgbp_df.empty:
            xgbp_df = xgbp_df.rename(columns={"y_pred": "y_pred_xgb_price"})
        
    
        # --- Merge sentiment ---
        pred_sent_df = pd.DataFrame()
        if not rf_df.empty and not xgb_df.empty:
            pred_sent_df = pd.merge(rf_df, xgb_df, on="timestamp", how="outer")
        else:
            pred_sent_df = rf_df if not rf_df.empty else xgb_df
        if not pred_sent_df.empty:
            pred_sent_df = pred_sent_df.sort_values("timestamp")

        # --- Merge price ---
        pred_price_df = pd.DataFrame()
        if not rfp_df.empty and not xgbp_df.empty:
            pred_price_df = pd.merge(rfp_df, xgbp_df, on="timestamp", how="outer")
        else:
            pred_price_df = rfp_df if not rfp_df.empty else xgbp_df
        if not pred_price_df.empty:
            pred_price_df = pred_price_df.sort_values("timestamp")
    


        if mode == "Regímenes (Halvings)":
            target_ts = st.session_state.get("__regime_target_ts__")
            regime    = st.session_state.get("__regime_name__")

            st.caption(f"Régimen: {regime}. Predicción dinámica disponible al pulsar 🚀 Ejecutar Predicción.")

            # Mostrar resultados dinámicos guardados (si existen)
            dyn_sent = st.session_state.get("__dyn_reg_result__")
            dyn_price = st.session_state.get("__dyn_reg_price_result__")

            # 1) Mostrar resultado dinámico si existe (guardado por el panel Regímenes)

            def _pretty_dyn(d: dict, label: str):
                """Devuelve un dict corto y 'lindo' para mostrar."""
                if not isinstance(d, dict):
                    return None
                pred = d.get("prediction", None)
                pred_lbl = None
                if pred in (0, 1):
                    pred_lbl = "SUBIDA" if int(pred) == 1 else "BAJADA"

                return {
                    "tipo": label,
                    "prediction": pred_lbl,
                    "confidence": d.get("confidence", None),
                    "proba_up": d.get("proba_up", None),
                    "mode_used": d.get("mode_used", None),
                    "model": d.get("model", None),
                    "resolution": d.get("resolution", None),
                    "timestamp": d.get("timestamp", None),
                    "msg": d.get("msg", None),
                }

            st.markdown("### 🔥 Predicción dinámica")

            c1, c2 = st.columns(2)

            with c1:
                st.write("**Sentimiento**")
                if isinstance(dyn_sent, dict) and dyn_sent.get("prediction") is not None:
                    st.write(_pretty_dyn(dyn_sent, "sentiment"))
                else:
                    st.info("Aún no hay predicción dinámica de sentimiento. Pulsa 🚀 en el sidebar para generarla.")

            with c2:
                st.write("**Precio**")
                if isinstance(dyn_price, dict) and dyn_price.get("prediction") is not None:
                    st.write(_pretty_dyn(dyn_price, "price"))
                else:
                    st.info("Aún no hay predicción dinámica de precio. Pulsa 🚀 en el sidebar para generarla.")

            # Debug detallado opcional
            if debug_mode:
                with st.expander("Detalles completos (debug)", expanded=False):
                    st.write({"dyn_sent_raw": dyn_sent, "dyn_price_raw": dyn_price})


            # 2) Mostrar predicciones históricas precomputadas alrededor del target
            if target_ts is None:
                st.info("Selecciona una fecha objetivo para ver predicciones cercanas.")
            else:
                lo, hi = target_ts - pd.Timedelta(days=7), target_ts + pd.Timedelta(days=7)

                rf_win  = rf_df[(rf_df["timestamp"] >= lo) & (rf_df["timestamp"] <= hi)].copy() if not rf_df.empty else pd.DataFrame()
                xgb_win = xgb_df[(xgb_df["timestamp"] >= lo) & (xgb_df["timestamp"] <= hi)].copy() if not xgb_df.empty else pd.DataFrame()

                st.markdown("### 📄 Predicciones históricas (CSV) — ventana ±7 días")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("RF (clasificación)")
                    if not rf_win.empty:
                        st.dataframe(rf_win.head(80), width="stretch")
                    else:
                        st.info("No hay datos RF en la ventana seleccionada.")
                with c2:
                    st.write("XGB (clasificación)")
                    if not xgb_win.empty:
                        st.dataframe(xgb_win.head(80), width="stretch")
                    else:
                        st.info("No hay datos XGB en la ventana seleccionada.")

        else:
            # ---------- Global ----------
            if rf_df.empty and xgb_df.empty:
                st.info("Aún no hay CSVs de predicción de sentimiento en data/processed/.")
            if rfp_df.empty and xgbp_df.empty:
                st.info(f"No hay CSVs de predicción de PRECIO para {interval} en data/processed/.")

            else:
                # Merge sentiment
                pred_sent_df = pd.DataFrame()
                if not rf_df.empty and not xgb_df.empty:
                    pred_sent_df = pd.merge(rf_df, xgb_df, on="timestamp", how="outer")
                else:
                    pred_sent_df = rf_df if not rf_df.empty else xgb_df

                if not pred_sent_df.empty:
                    pred_sent_df = pred_sent_df.sort_values("timestamp")

                # Merge price
                pred_price_df = pd.DataFrame()
                if not rfp_df.empty and not xgbp_df.empty:
                    pred_price_df = pd.merge(rfp_df, xgbp_df, on="timestamp", how="outer")
                else:
                    pred_price_df = rfp_df if not rfp_df.empty else xgbp_df

                if not pred_price_df.empty:
                    pred_price_df = pred_price_df.sort_values("timestamp")

                
                # Ventana sentimiento (solo si existe timestamp)
                if (pred_sent_df is None) or pred_sent_df.empty or ("timestamp" not in pred_sent_df.columns):
                    pred_window = pd.DataFrame()
                else:
                    pred_window = pred_sent_df[
                        (pred_sent_df["timestamp"] >= start_ts) & (pred_sent_df["timestamp"] <= end_ts)
                    ].copy()

                if pred_window.empty:
                    st.warning("Hay predicciones, pero no hay registros dentro del rango de fechas seleccionado.")
                else:
                    st.markdown(f"### ⏭️ Predicción — siguiente {interval} (+{horizon_hours}h)")

                    last_series = price_window.dropna(subset=["price"])
                    if last_series.empty:
                        st.info("No hay precio disponible para calcular el timestamp objetivo (t+1).")
                    else:
                        last_seen_ts = last_series["timestamp"].max()
                        target_ts_g  = last_seen_ts + pd.Timedelta(hours=horizon_hours)
                        st.caption(f"Base (último dato): {last_seen_ts:%Y-%m-%d %H:%M} UTC → Objetivo: {target_ts_g:%Y-%m-%d %H:%M} UTC")

                        row_next = pred_window[pred_window["timestamp"] == target_ts_g]
                        if row_next.empty:
                            st.info(f"Aún no existe una fila de predicción exactamente para el siguiente {interval} (t+1).")
                        else:
                            rf_val  = row_next["y_pred_rf"].iloc[0] if "y_pred_rf" in row_next.columns else None
                            xgb_val = row_next["y_pred_xgb"].iloc[0] if "y_pred_xgb" in row_next.columns else None

                            # Mostrar como clase (SUBIDA/BAJADA), no como número flotante
                            def _cls_label(v):
                                if v is None or pd.isna(v): return "N/A"
                                return "SUBIDA" if int(v) == 1 else "BAJADA"

                            c1, c2, c3 = st.columns(3)
                            c1.metric("t+1 timestamp", f"{target_ts_g:%Y-%m-%d %H:%M} UTC")
                            c2.metric("RF pred (t+1)", _cls_label(rf_val))
                            c3.metric("XGB pred (t+1)", _cls_label(xgb_val))

                            # Consenso (ambos son 0/1)
                            directions = []
                            if rf_val is not None and pd.notna(rf_val):
                                directions.append(int(rf_val))
                            if xgb_val is not None and pd.notna(xgb_val):
                                directions.append(int(xgb_val))

                            if len(directions) == 2 and directions[0] == directions[1]:
                                st.success("✅ Ambos modelos coinciden en la dirección (mayor confianza).")
                            elif len(directions) == 2 and directions[0] != directions[1]:
                                st.warning("⚠️ Los modelos discrepan en dirección (señal mixta).")
                            else:
                                st.info("ℹ️ Solo hay salida de un modelo disponible por ahora.")

                    # Plot & table (clasificación + y_true si existe)
                    st.markdown("### 📈 Observed vs Predicted (clasificación)")
                    plot_df = pred_window.set_index("timestamp")
                    cols = []
                    if "y_true" in plot_df.columns: cols.append("y_true")
                    if "y_pred_rf" in plot_df.columns: cols.append("y_pred_rf")
                    if "y_pred_xgb" in plot_df.columns: cols.append("y_pred_xgb")
                    if cols:
                        st.line_chart(plot_df[cols])
                    else:
                        st.info("No se encontraron columnas de predicción para graficar.")

                    st.markdown("### 📄 Predicciones (tabla)")
                    st.dataframe(pred_window.head(200), width='stretch')
                    csv_bytes = pred_window.to_csv(index=False).encode("utf-8")
                    st.download_button("Descargar predicciones (CSV)", csv_bytes, "predictions_filtered.csv", "text/csv")


    # --------- Tab Methodología (EDA) ---------
    with tab_method:
        st.subheader("Metodología (resumen)")
        df = price_window.dropna(subset=["price"]).copy().sort_values("timestamp")

        if df.empty or "timestamp" not in df.columns or "price" not in df.columns:
            st.info("No hay datos suficientes para mostrar metodología en el rango/regimen.")
        else:
            s = df.set_index("timestamp")["price"].astype(float)
            st.markdown("### Serie de precio")
            st.line_chart(s)

            st.markdown("### Retornos (log) y volatilidad")
            log_s = s.apply(lambda x: math.log(x) if x > 0 else float("nan"))
            rets = log_s.diff().dropna()
            st.caption("Retornos log (%)")
            st.line_chart((rets * 100).rename("log_return_%"))
            st.caption("Volatilidad rolling (24 pasos) sobre retornos log (√24).")
            vol = rets.rolling(24).std() * (24 ** 0.5)
            st.line_chart(vol.dropna().rename("vol_24"))

            # EDA de Sentimiento solo en Global
            if mode == "Global" and not news_window.empty:
                st.divider()
                st.markdown("### EDA de Sentimiento (Global)")
                if "sentiment_score" in news_window.columns:
                    snt = pd.to_numeric(news_window["sentiment_score"], errors="coerce").dropna()
                    st.write({
                        "mean": float(snt.mean()),
                        "median": float(snt.median()),
                        "std": float(snt.std()),
                        "pos(>0.05)": int((snt > 0.05).sum()),
                        "neg(<-0.05)": int((snt < -0.05).sum()),
                        "n": int(len(snt)),
                    })
                else:
                    st.info("No se encontró columna 'sentiment_score' en noticias.")

            st.divider()
            st.markdown("### Hallazgos (resumen)")
            st.write(
                "- La serie de precio requiere diferenciación (d≈1) para estacionariedad.\n"
                "- Estacionalidad débil; ARIMA/SARIMA es baseline razonable.\n"
                "- En modo Regímenes, cada modelo se entrena con su subperiodo (estacionalidad no compartida)."
            )

    # --------- Tab Detalles ---------
    with tab_details:
        st.subheader("Resumen de datos")
        fuente = "raw" if "raw" in str(price_path).lower() else "processed"
        rows_news = int(len(news_window)) if mode == "Global" else 0
        sources = int(news_window["source"].nunique()) if (mode == "Global" and not news_window.empty and "source" in news_window.columns) else 0

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
    options=["Home", "Overview", "Dashboard", "Recomendaciones"],
    icons=["house", "card-text", "graph-up", "lightbulb"],
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
    render_app()
elif selected == "Recomendaciones":
    render_recommendations()

