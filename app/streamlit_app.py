import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import streamlit as st
import pandas as pd

from src.frontend.data_loader import (
    load_prices, build_ohlc, load_news, filter_news_timewindow,
    default_relevance_filter, quick_sentiment_score, load_prediction_csv
)
from src.frontend.ui_components import candlestick_chart, line_with_ma

PRICE_DEFAULT_PATH = 'data/raw/prices_raw_test.csv'
NEWS_DEFAULT_PATH  = 'data/raw/news_raw_test.csv'

# Cuando empecemos a ocupar los modelos, los datos se deberan guardar en 'processed'
RF_PRED_PATH = 'data/processed/predictions_rf.csv'
ARIMA_PRED_PATH = 'data/processed/predictions_arima.csv'


st.set_page_config(page_title='BTC Predictor', page_icon='📈', layout='wide')

st.markdown(
    """ <style>
    .main { background-color: #0b1120; color: #e5e7eb; }
    .stMetric { background-color: #020617 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('BTC Predictor')
st.caption('Panel para explorar precio, noticias y una señal agregada basada en sentimiento.')

# ------------------ Sidebar ------------------
st.sidebar.header('Controles')
st.sidebar.title('Panel de control')
st.sidebar.write('Ajusta el rango de fechas y los filtros para ver cómo cambia la señal.')

# Carga de datos desde data/raw
prices_df = load_prices(PRICE_DEFAULT_PATH)
news_df = load_news(NEWS_DEFAULT_PATH)

min_ts = min(prices_df['timestamp'].min(), news_df['published_at'].min())
max_ts = max(prices_df['timestamp'].max(), news_df['published_at'].max())

start_date = st.sidebar.date_input('Fecha inicio', value=min_ts.date())
end_date = st.sidebar.date_input('Fecha fin', value=max_ts.date())

start_ts = pd.Timestamp(start_date).tz_localize('UTC')
end_ts = pd.Timestamp(end_date).tz_localize('UTC') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

interval = st.sidebar.selectbox('Granularidad', ['1h','4h','1d'], index=0)
relevance = st.sidebar.checkbox('Filtrar noticias relevantes', value=True)
horizon = st.sidebar.slider('Horizonte de predicción (horas)', 1, 24, 1)
refresh = st.sidebar.button('Actualizar')

# ------------------ Prepare data ------------------
price_window = prices_df[(prices_df['timestamp'] >= start_ts) & (prices_df['timestamp'] <= end_ts)].copy()
news_window = filter_news_timewindow(news_df, start_ts, end_ts)

if relevance:
    news_window = default_relevance_filter(news_window)

price_pack = build_ohlc(price_window, interval=interval, min_points_per_candle=2)

# ------------------ Layout ------------------
col_price, col_stats = st.columns([2, 1])

with col_price:
    st.subheader('Precio BTC')

    if price_pack.mode in ('ohlc', 'computed_ohlc') and not price_pack.ohlc.empty:
        chart = candlestick_chart(price_pack.ohlc, ts_col='timestamp').properties(height=380)
        st.altair_chart(chart, use_container_width=True)
    else:
        non_null = price_window.dropna(subset=['price']).copy()
        if not non_null.empty:
            chart = line_with_ma(non_null, ts_col='timestamp', price_col='price', ma_window=24).properties(height=380)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning('No hay datos de precio disponibles en el rango seleccionado.')

with col_stats:
    st.subheader('Sentimiento del mercado')

    if news_window.empty:
        sentiment_score, sentiment_label, pos_hits, neg_hits = 0.0, 'Neutral', 0, 0
    else:
        sentiment_score, sentiment_label, pos_hits, neg_hits = quick_sentiment_score(news_window)
    
    # Probabilidad en 0–100 a partir del score (-1 a 1)
    prob = int((sentiment_score + 1) * 50)

    st.metric('Sentimiento agregado', sentiment_label, f'{sentiment_score:+.2f}')
    st.progress(prob)
    st.caption(f'Probabilidad (escala interna): {prob}%')

    st.subheader('Señal final')

    if sentiment_score > 0.10:
        final_label, color, conf = 'SUBIDA', 'green', prob
    elif sentiment_score < -0.10:
        final_label, color, conf = 'BAJADA', 'red', prob
    else:
        final_label, color, conf = 'ESCENARIO NEUTRAL', 'gray', prob

    st.markdown(f"<h3 style='color:{color}; margin:0'>{final_label}</h3>", unsafe_allow_html=True)
    st.progress(conf)

st.divider()

tab1, tab2, tab3 = st.tabs(['Noticias', 'Predicción', 'Datos'])

with tab1:
    st.subheader('Noticias filtradas')
    st.caption(f'Registros: {len(news_window)}')
    if not news_window.empty:
        st.dataframe(news_window[['published_at','source','axis','title']].head(200), use_container_width=True)
        csv_bytes = news_window.to_csv(index=False).encode('utf-8')
        st.download_button('Descargar CSV', csv_bytes, 'news_filtered.csv', 'text/csv')
    else:
        st.info('No hay noticias para mostrar.')

with tab2:
    st.subheader("Predicción")
    st.caption("Se integran dos salidas (exportadas a CSV) - Nodelos Random Forest y ARIMA/SARIMA")

    # 1) Cargar ambos archivos (si existen)
    rf_df = load_prediction_csv(RF_PRED_PATH, "y_pred_rf")
    arima_df = load_prediction_csv(ARIMA_PRED_PATH, "y_pred_arima")

    # 2) Si no hay nada aún, deja el placeholder (como ahora)
    if rf_df.empty and arima_df.empty:
        st.info(
            "En esta versión, la pestaña es solo informativa. "
            "Cuando existan los CSV en data/processed/, se mostrará aquí las curvas y la dirección final."
        )
    else:
        # 3) Unir por timestamp (outer para no perder filas)
        if not rf_df.empty and not arima_df.empty:
            pred_df = pd.merge(rf_df, arima_df, on="timestamp", how="outer")

            # Consolidar y_true si viene en ambos
            if "y_true_x" in pred_df.columns or "y_true_y" in pred_df.columns:
                pred_df["y_true"] = pred_df.get("y_true_x")
                if "y_true_y" in pred_df.columns:
                    pred_df["y_true"] = pred_df["y_true"].fillna(pred_df["y_true_y"])
                pred_df = pred_df.drop(columns=[c for c in ["y_true_x", "y_true_y"] if c in pred_df.columns])
        else:
            pred_df = rf_df if not rf_df.empty else arima_df

        pred_df = pred_df.sort_values("timestamp")

        # 4) Filtrar por el mismo rango seleccionado en el sidebar
        pred_window = pred_df[(pred_df["timestamp"] >= start_ts) & (pred_df["timestamp"] <= end_ts)].copy()

        if pred_window.empty:
            st.warning("Hay predicciones, pero no hay registros dentro del rango de fechas seleccionado.")
        else:
            # --- Next hour (t+1) ---
            st.markdown("### ⏭️ Predicción próxima hora (t+1)")

            last_price_series = price_window.dropna(subset=["price"])
            if last_price_series.empty:
                st.info("No hay precio disponible para calcular el timestamp objetivo (t+1).")
            else:
                last_seen_ts = last_price_series["timestamp"].max()
                target_ts = last_seen_ts + pd.Timedelta(hours=1)

                row_next = pred_df[pred_df["timestamp"] == target_ts]

                if row_next.empty:
                    st.info("Aún no existe una fila de predicción exactamente para la próxima hora (t+1).")
                else:
                    last_price = float(last_price_series.iloc[-1]["price"])

                    rf_val = row_next["y_pred_rf"].iloc[0] if "y_pred_rf" in row_next.columns else None
                    ar_val = row_next["y_pred_arima"].iloc[0] if "y_pred_arima" in row_next.columns else None

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Último precio observado (t)", f"{last_price:,.2f}")
                    c2.metric("RF pred (t+1)", f"{rf_val:,.2f}" if rf_val is not None and pd.notna(rf_val) else "N/A")
                    c3.metric("ARIMA/SARIMA pred (t+1)", f"{ar_val:,.2f}" if ar_val is not None and pd.notna(ar_val) else "N/A")

                    # Agreement simple (no inventamos un modelo final)
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

            # --- Plot: observed vs predicted (o solo pred) ---
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
            st.dataframe(pred_window.head(200), use_container_width=True)

            csv_bytes = pred_window.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar predicciones (CSV)", csv_bytes, "predictions_filtered.csv", "text/csv")


with tab3:
    st.subheader('Resumen de datos')

    st.write({
        'rows_prices': int(len(price_window)),
        'rows_news': int(len(news_window)),
        'sources_unicas': int(news_window['source'].nunique()) if not news_window.empty else 0
    })
