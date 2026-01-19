from datetime import datetime, timezone
import sys
import os
from pathlib import Path
import base64

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

ROOT_PATH = Path(ROOT)
ASSETS_DIR = ROOT_PATH / "assets"
RAW_DIR = ROOT_PATH / "data" / "raw"
PROCESSED_DIR = ROOT_PATH / "data" / "processed"

from src.frontend.data_loader import (
    load_prices, build_ohlc, load_news, filter_news_timewindow,
    default_relevance_filter, quick_sentiment_score, load_prediction_csv
)
from src.frontend.ui_components import candlestick_chart, line_with_ma


# --- Paths: raw (test) ---
PRICE_TEST_PATH = RAW_DIR / "prices_raw_test.csv"
NEWS_TEST_PATH  = RAW_DIR / "news_raw_test.csv"

# --- Paths: processed (ideal/pipeline) ---
PRICE_PROCESSED_PATH = PROCESSED_DIR / "prices_raw.csv"
NEWS_PROCESSED_PATH  = PROCESSED_DIR / "news_raw.csv"

# --- Predictions (pipeline outputs) ---
RF_PRED_PATH = PROCESSED_DIR / "predictions_rf.csv"
ARIMA_PRED_PATH = PROCESSED_DIR / "predictions_arima.csv"


# ------------------ Helpers ------------------
def _asset_to_data_uri(path: Path) -> str | None:
    """Convierte imagen local a data URI base64 para usar en CSS. Si no existe, None."""
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

@st.cache_data(show_spinner=False)
def _load_prices_cached(path_str: str) -> pd.DataFrame:
    return load_prices(path_str)

@st.cache_data(show_spinner=False)
def _load_news_cached(path_str: str) -> pd.DataFrame:
    return load_news(path_str)

def _pick_data_paths(prefer_processed: bool) -> tuple[Path, Path]:
    """Elige paths para precios/noticias según exista processed o no."""
    if prefer_processed and PRICE_PROCESSED_PATH.exists() and NEWS_PROCESSED_PATH.exists():
        return PRICE_PROCESSED_PATH, NEWS_PROCESSED_PATH
    # fallback a test
    return PRICE_TEST_PATH, NEWS_TEST_PATH

# ------------------ Streamlit config ------------------
st.set_page_config(page_title='BTC Predictor', page_icon='📈', layout='wide')

# ------------------ Load Bootstrap Icons ------------------
st.markdown(
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
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
      min-height: 72vh;
      padding: 4rem 2rem;
      border-radius: 16px;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      background-color: #0F1015;
      color: #e5e7eb;
      position: relative; overflow: hidden;
    }}
    .hero::before{{
      content:"";
      position:absolute; inset:0;
      background-repeat: repeat;
      background-size: 220px;
      background-position: 0 0;
      opacity: 0.08;
      filter: grayscale(100%);
      background-image: url('{btc_bg_uri if btc_bg_uri else ""}');
    }}
    .hero h1{{ margin:0 0 .5rem 0; font-size: clamp(2.2rem, 4vw, 3.4rem); z-index:1; }}
    .hero .tagline{{ color:#94a3b8; font-size: 1.05rem; text-align:center; max-width: 900px; z-index:1; }}
    .hero .cta{{ margin-top: 1.5rem; z-index:1; }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Navegación: helper para saltar de página desde botones ---
NAV_INDEX = {"Home": 0, "Overview": 1, "Dashboard": 2}

def go_to(page_name: str):
    # Guardamos el índice que queremos seleccionar en el navbar
    st.session_state["nav_manual_select"] = NAV_INDEX.get(page_name, 0)

# ------------------ Views ------------------
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

OVERVIEW_MD = """
## BTC Predictor

En un mercado altamente volátil, el precio refleja el resultado de la actividad del mercado, pero no siempre explica sus causas.  
Noticias y eventos pueden influir en el comportamiento de los inversores y amplificar movimientos.  
Este proyecto integra **datos cuantitativos** (precio e indicadores) con **datos cualitativos** (sentimiento) para aportar contexto a la predicción.

---

### Resumen
**BTC Predictor** es un modelo predictivo del precio de Bitcoin que combina **análisis técnico** (series temporales e indicadores) con **análisis de sentimiento** (noticias y eventos).  
El objetivo es ofrecer una señal más completa para la toma de decisiones:  
**El análisis OHLC muestra el “qué” (movimiento del precio)** y el **sentimiento aporta el “por qué” (drivers emocionales)**.
"""


def render_overview():
    st.header("Overview")
    st.markdown(OVERVIEW_MD)
    st.button("Ir a la App →", type="primary", on_click=go_to, args=("Dashboard",))
    st.markdown("</div></div>", unsafe_allow_html=True)


def render_app():
    st.title('BTC Predictor')
    st.caption("Dashboard para analizar precio, noticias y sentimiento del mercado en Bitcoin ₿.")
    
    # --- Guardar hora de apertura (una sola vez por sesión) ---
    if "opened_at_utc" not in st.session_state:
        st.session_state["opened_at_utc"] = datetime.now(timezone.utc)

    # ------------------ Sidebar (solo en App) ------------------
    st.sidebar.title("Panel de control")
    st.sidebar.write("Ajusta el rango de fechas y los filtros para ver cómo cambia la señal.")

    prefer_processed = PRICE_PROCESSED_PATH.exists() and NEWS_PROCESSED_PATH.exists()
    price_path, news_path = _pick_data_paths(prefer_processed)

    prices_df = _load_prices_cached(str(price_path))
    news_df   = _load_news_cached(str(news_path))

    if prices_df.empty or news_df.empty:
        st.error("No se pudieron cargar precios o noticias (revisa data/raw y/o data/processed).")
        return

    min_ts = min(prices_df["timestamp"].min(), news_df["published_at"].min())
    max_ts = max(prices_df["timestamp"].max(), news_df["published_at"].max())

    # Para no abrir siemprer con hisórico - set Defaults
    default_end = max_ts.date()
    default_start = (max_ts - pd.Timedelta(days=90)).date() if hasattr(max_ts, "date") else min_ts.date()

    # --- Widgets SIN form => rerun inmediato al cambiar ---
    start_date = st.sidebar.date_input(
        "Fecha inicio",
        value=st.session_state.get("start_date", default_start),
        key="start_date",
    )

    end_date = st.sidebar.date_input(
        "Fecha fin",
        value=st.session_state.get("end_date", default_end),
        key="end_date",
    )

    interval = st.sidebar.selectbox(
        "Granularidad",
        ["1h", "4h", "1d"],
        index=["1h", "4h", "1d"].index(st.session_state.get("interval", "1h")),
        key="interval",
    )

    relevance = st.sidebar.checkbox(
        "Filtrar noticias relevantes",
        value=st.session_state.get("relevance", True),
        key="relevance",
    )

    horizon = st.sidebar.slider(
        "Horizonte de predicción (horas)",
        1, 24,
        value=int(st.session_state.get("horizon", 1)),
        key="horizon",
    )

    # --- Botón “Actualizar datos” al final: SOLO para refrescar CSV/caché ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Actualizar datos", type="secondary"):
        st.cache_data.clear()
        st.rerun()


    # --- Convertir a timestamps (UTC) ---
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


    # --- Estado del dashboard (cliente-friendly) ---
    opened = st.session_state["opened_at_utc"]
    st.sidebar.caption(f"Abierto: {opened:%Y-%m-%d %H:%M} UTC")
    st.sidebar.caption(f"Datos hasta: {max_ts:%Y-%m-%d %H:%M} UTC")

    # ------------------ Prepare data ------------------
    price_window = prices_df[(prices_df["timestamp"] >= start_ts) & (prices_df["timestamp"] <= end_ts)].copy()
    news_window = filter_news_timewindow(news_df, start_ts, end_ts)

    if relevance:
        news_window = default_relevance_filter(news_window)

    price_pack = build_ohlc(price_window, interval=interval, min_points_per_candle=2)

    # ------------------ Layout (Precio + Sentimiento) ------------------
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
                st.warning("No hay datos de precio disponibles en el rango seleccionado.")

    with col_stats:
        st.subheader("Sentimiento del mercado")

        if news_window.empty:
            sentiment_score, sentiment_label, pos_hits, neg_hits = 0.0, "Neutral", 0, 0
        else:
            sentiment_score, sentiment_label, pos_hits, neg_hits = quick_sentiment_score(news_window)

        prob = int((sentiment_score + 1) * 50)

        st.metric("Sentimiento agregado", sentiment_label, f"{sentiment_score:+.2f}")
        st.progress(prob)
        st.caption(f"Probabilidad (escala interna): {prob}%")
        st.caption(f"Hits → Positivos: {pos_hits} | Negativos: {neg_hits}")

        st.subheader("Señal final")
        if sentiment_score > 0.10:
            final_label, color, conf = "SUBIDA", "green", prob
        elif sentiment_score < -0.10:
            final_label, color, conf = "BAJADA", "red", prob
        else:
            final_label, color, conf = "ESCENARIO NEUTRAL", "gray", prob

        st.markdown(f"<h3 style='color:{color}; margin:0'>{final_label}</h3>", unsafe_allow_html=True)
        st.progress(conf)

    st.divider()

    # ------------------ Tabs ------------------
    tab_signal, tab_context, tab_method, tab_details = st.tabs(["Señal & Predicción", "Contexto", "Metodología", "Detalles"])

    with tab_context:
        st.subheader("Noticias filtradas")
        st.caption(f"Registros: {len(news_window)}")

        if not news_window.empty:
            st.dataframe(news_window[["published_at", "source", "axis", "title"]].head(200), width='stretch')
            csv_bytes = news_window.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv_bytes, "news_filtered.csv", "text/csv")
        else:
            st.info("No hay noticias para mostrar.")

    with tab_signal:
        st.subheader("Predicción")
        st.caption("Se integran dos salidas (exportadas a CSV) - Modelos Random Forest y ARIMA/SARIMA")

        rf_df = load_prediction_csv(str(RF_PRED_PATH), "y_pred_rf")
        arima_df = load_prediction_csv(str(ARIMA_PRED_PATH), "y_pred_arima")

        if rf_df.empty and arima_df.empty:
            st.info(
                "En esta versión, la pestaña es informativa. "
                "Cuando existan los CSV en data/processed/, se mostrará aquí las curvas y la dirección final."
            )
        else:
            # Merge outer por timestamp
            if not rf_df.empty and not arima_df.empty:
                pred_df = pd.merge(rf_df, arima_df, on="timestamp", how="outer")

                # Consolidar y_true si viene duplicado
                if "y_true_x" in pred_df.columns or "y_true_y" in pred_df.columns:
                    pred_df["y_true"] = pred_df.get("y_true_x")
                    if "y_true_y" in pred_df.columns:
                        pred_df["y_true"] = pred_df["y_true"].fillna(pred_df["y_true_y"])
                    pred_df = pred_df.drop(columns=[c for c in ["y_true_x", "y_true_y"] if c in pred_df.columns])
            else:
                pred_df = rf_df if not rf_df.empty else arima_df

            pred_df = pred_df.sort_values("timestamp")

            # Filtrar mismo rango
            pred_window = pred_df[(pred_df["timestamp"] >= start_ts) & (pred_df["timestamp"] <= end_ts)].copy()

            if pred_window.empty:
                st.warning("Hay predicciones, pero no hay registros dentro del rango de fechas seleccionado.")
            else:
                # --- Next horizon (t+horizon) ---
                st.markdown(f"### ⏭️ Predicción horizonte (t+{horizon}h)")

                last_price_series = price_window.dropna(subset=["price"])
                if last_price_series.empty:
                    st.info("No hay precio disponible para calcular el timestamp objetivo (t+h).")
                else:
                    last_seen_ts = last_price_series["timestamp"].max()
                    target_ts = last_seen_ts + pd.Timedelta(hours=horizon)

                    st.caption(f"Base (último dato): {last_seen_ts:%Y-%m-%d %H:%M} UTC → Objetivo: {target_ts:%Y-%m-%d %H:%M} UTC")

                    row_next = pred_df[pred_df["timestamp"] == target_ts]

                    if row_next.empty:
                        st.info(f"Aún no existe una fila de predicción exactamente para t+{horizon}h.")
                    else:
                        last_price = float(last_price_series.iloc[-1]["price"])

                        rf_val = row_next["y_pred_rf"].iloc[0] if "y_pred_rf" in row_next.columns else None
                        ar_val = row_next["y_pred_arima"].iloc[0] if "y_pred_arima" in row_next.columns else None

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Último precio observado (t)", f"{last_price:,.2f}")
                        c2.metric(f"RF pred (t+{horizon}h)", f"{rf_val:,.2f}" if rf_val is not None and pd.notna(rf_val) else "N/A")
                        c3.metric(f"ARIMA/SARIMA pred (t+{horizon}h)", f"{ar_val:,.2f}" if ar_val is not None and pd.notna(ar_val) else "N/A")

                        directions = []
                        if rf_val is not None and pd.notna(rf_val):
                            directions.append("up" if rf_val > last_price else "down" if rf_val < last_price else "flat")
                        if ar_val is not None and pd.notna(ar_val):
                            directions.append("up" if ar_val > last_price else "down" if ar_val < last_price else "flat")

                        if len(directions) >= 2 and directions[0] == directions[1] and directions[0] != "flat":
                            st.success("✅ Ambos modelos coinciden en la dirección (mayor confianza).")
                        elif len(directions) >= 2 and directions[0] != directions[1]:
                            st.warning("⚠️ Los modelos discrepan en dirección (señal mixta).")
                        else:
                            st.info("ℹ️ Solo hay salida de un modelo disponible por ahora.")

                # --- Plot ---
                st.markdown("### 📈 Observed vs Predicted")
                plot_df = pred_window.set_index("timestamp")
                cols = []
                if "y_true" in plot_df.columns:
                    cols.append("y_true")
                if "y_pred_rf" in plot_df.columns:
                    cols.append("y_pred_rf")
                if "y_pred_arima" in plot_df.columns:
                    cols.append("y_pred_arima")

                if not cols:
                    st.info("No se encontraron columnas de predicción para graficar.")
                else:
                    st.line_chart(plot_df[cols])

                # --- Table + download ---
                st.markdown("### 📄 Predicciones (tabla)")
                st.dataframe(pred_window.head(200), width='stretch')

                csv_bytes = pred_window.to_csv(index=False).encode("utf-8")
                st.download_button("Descargar predicciones (CSV)", csv_bytes, "predictions_filtered.csv", "text/csv")

    with tab_details:
        st.subheader("Resumen de datos")
        st.write({
            "fuente": "processed" if (price_path == PRICE_PROCESSED_PATH) else "raw(test)",
            "rows_prices": int(len(price_window)),
            "rows_news": int(len(news_window)),
            "sources_unicas": int(news_window["source"].nunique()) if not news_window.empty else 0
        })


    with tab_method:
        st.subheader("Metodología (resumen)")

        if price_window.empty or "timestamp" not in price_window.columns or "price" not in price_window.columns:
            st.info("No hay datos suficientes para mostrar metodología en el rango seleccionado.")
        else:
            df = price_window.dropna(subset=["price"]).copy().sort_values("timestamp")
            s = df.set_index("timestamp")["price"].astype(float)

            st.markdown("### Serie de precio")
            st.line_chart(s)

            st.markdown("### Retornos (log) y volatilidad")
            # log-returns sin numpy
            log_s = s.apply(lambda x: math.log(x) if x > 0 else float("nan"))
            rets = log_s.diff().dropna()

            st.caption("Retornos log (%) — visión estándar en finanzas.")
            st.line_chart((rets * 100).rename("log_return_%"))

            st.caption("Volatilidad rolling (24 pasos) sobre retornos log.")
            vol = rets.rolling(24).std() * (24 ** 0.5)
            st.line_chart(vol.dropna().rename("vol_24"))

            st.divider()
            st.markdown("### Hallazgos del EDA (Notebook)")
            st.write(
                "- La serie requiere diferenciación para estacionariedad (d≈1 en el análisis del notebook).\n"
                "- No se detecta estacionalidad fuerte (en el notebook se considera débil y se recomienda ARIMA baseline).\n"
                "- Se sugiere ARIMA como baseline y comparar con modelos alternativos."
            )

        nb_path = ROOT_PATH / "notebooks" / "eda_time_series.ipynb"
        if nb_path.exists():
            st.caption("Notebook completo disponible en: `notebooks/eda_time_series.ipynb`")
        else:
            st.caption("Notebook se integrará en `main` como: `notebooks/eda_time_series.ipynb`")

# ------------------ Top nav (streamlit-option-menu) ------------------

manual = st.session_state.get("nav_manual_select", None)

selected = option_menu(
    menu_title=None,
    options=["Home", "Overview", "Dashboard"],
    icons=["house", "card-text", "graph-up"],   # Bootstrap icon names
    default_index=0 if manual is None else manual,
    orientation="horizontal",
    manual_select=manual,
    styles={
        "container": {
            "padding": "0.2rem 0.2rem",
            "background-color": "#0b1120",
        },
        "icon": {"color": "#e5e7eb", "font-size": "16px"},
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px 6px",
            "color": "#94a3b8",
            "border-radius": "10px",
            "padding": "6px 12px",
        },
        "nav-link-selected": {
            "background-color": "#111827",
            "color": "#e5e7eb",
        },
    },
)

# (Opcional) Oculta sidebar en Home/Overview para look "presentación"
if selected in ("Home", "Overview"):
    st.markdown("<style>section[data-testid='stSidebar']{display:none;}</style>", unsafe_allow_html=True)


# Limpia manual_select después de aplicarlo (para no “forzar” siempre)
if manual is not None:
    st.session_state["nav_manual_select"] = None

if selected == "Home":
    render_home()
elif selected == "Overview":
    render_overview()
else:
    render_app()
