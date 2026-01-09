import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import streamlit as st
import pandas as pd

from src.frontend.data_loader import (
    load_prices, build_ohlc, load_news, filter_news_timewindow,
    default_relevance_filter, quick_sentiment_score
)
from src.frontend.ui_components import candlestick_chart, line_with_ma

PRICE_DEFAULT_PATH = 'data/raw/prices_raw_test.csv'
NEWS_DEFAULT_PATH  = 'data/raw/news_raw_test.csv'

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
    st.subheader('Predicción')
    st.caption('Aquí se integrará el archivo de predicciones cuando esté disponible.')
    st.info( 'En esta versión, la pestaña es solo informativa. ' 'Cuando existan los CSV de predicciones (por ejemplo en data/processed/), ' 'se mostrarán aquí las curvas de precio observado vs. precio predicho y la dirección final.' )

with tab3:
    st.subheader('Resumen de datos')

    st.write({
        'rows_prices': int(len(price_window)),
        'rows_news': int(len(news_window)),
        'sources_unicas': int(news_window['source'].nunique()) if not news_window.empty else 0
    })
