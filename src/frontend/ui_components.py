"""
Componentes reutilizables de visualización para Streamlit.
Incluye gráficos de velas (OHLC) y líneas con media móvil.
"""

import altair as alt
import pandas as pd


def candlestick_chart(df_ohlc: pd.DataFrame, ts_col: str = 'timestamp'):
    """
    Genera un gráfico de velas OHLC con Altair.
    Requiere columnas: open, high, low, close.
    """
    base = alt.Chart(df_ohlc).encode(x=f'{ts_col}:T')

    # ----- Línea vertical high-low -----
    rule = base.mark_rule().encode(
        y='low:Q',
        y2='high:Q',
        tooltip=[f'{ts_col}:T','open:Q','high:Q','low:Q','close:Q']
    )

    # ----- Cuerpo de la vela -----
    bar = base.mark_bar().encode(
        y='open:Q',
        y2='close:Q',
        color=alt.condition(
            'datum.open <= datum.close',
            alt.value('#22a884'),  # verde
            alt.value("#b91e0d")   # rojo
        )
    )

    return rule + bar


def line_with_ma(df: pd.DataFrame, ts_col: str = 'timestamp', price_col: str = 'price', ma_window: int = 24):
    """
    Gráfico de línea con media móvil y degradado sutil.
    - price_col: columna de precios
    - ma_window: tamaño de la ventana para la media móvil
    """
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
