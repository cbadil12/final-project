# app/09_backtest.py
# ===============================
# Simple backtest for classification model predictions
# Simulates long/flat strategy with basic transaction costs
# Outputs equity curve, key performance metrics
# ===============================

import os
import logging

import pandas as pd
import numpy as np
import xgboost as xgb

# ===============================
# CONFIGURATION
# ===============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURES_PATH = 'data/processed/features_with_prices_1h.csv'  # cambiar a 4h si quieres
MODEL_PATH = 'data/processed/xgb_clf_1h.json'

THRESHOLD = 0.52          # Puedes ajustar (ej: >0.52 para ser más selectivo)
TX_COST = 0.0008          # 0.08% por trade (típico Binance spot + slippage aproximado)
MIN_PROBA_DIFF = 0.04     # Diferencia mínima para entrar/salir (opcional)

# ===============================
# LOAD DATA & MODEL
# ===============================
def load_data_and_model():
    if not os.path.exists(FEATURES_PATH):
        logging.error(f"Features not found: {FEATURES_PATH}")
        return None
    
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    # Features
    drop_cols = ['open', 'high', 'low', 'close', 'volume', 'return']
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Needed for simulation
    prices = df['close']
    returns = df['return']
    
    # Load model
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    dmatrix = xgb.DMatrix(X)
    
    proba = model.predict(dmatrix)
    
    logging.info(f"Loaded data & model. {len(df)} candles, proba shape: {proba.shape}")
    return df, proba, prices, returns

# ===============================
# BACKTEST SIMULATION
# ===============================
def run_backtest(df, proba, prices, returns):
    results = pd.DataFrame(index=df.index)
    results['close'] = prices
    results['return'] = returns
    results['prob_up'] = proba
    
    # Decision: 1 = long, 0 = flat
    results['pred'] = (results['prob_up'] > THRESHOLD).astype(int)
    
    # Signal changes (para calcular trades)
    results['position'] = results['pred']
    results['trade'] = results['position'].diff().fillna(0).abs()  # 1 = enter/exit
    
    # Strategy returns with transaction costs
    results['strat_return'] = 0.0
    
    # Apply costs only when entering or exiting a position
    results['strat_return'] = results['position'].shift(1) * results['return']
    results['strat_return'] -= results['trade'] * TX_COST
    
    # Cumulative performance
    results['equity'] = (1 + results['strat_return'].fillna(0)).cumprod()
    results['buy_hold'] = (1 + results['return'].fillna(0)).cumprod()
    
    # Key metrics
    total_return = results['equity'].iloc[-1] - 1
    bh_return = results['buy_hold'].iloc[-1] - 1
    
    n_trades = results['trade'].sum() / 2  # cada entrada+salida = 1 trade completo
    win_rate = (results['strat_return'][results['strat_return'] != 0] > 0).mean()
    
    avg_return_per_trade = results['strat_return'][results['trade'].shift(-1) == 1].mean()
    
    # Sharpe (annualizado aproximado para 1h: ~8760 periodos/año)
    strat_ret_series = results['strat_return'].fillna(0)
    sharpe = 0
    if strat_ret_series.std() > 0:
        sharpe = strat_ret_series.mean() / strat_ret_series.std() * np.sqrt(8760)
    
    # Max drawdown
    rolling_max = results['equity'].cummax()
    drawdown = (results['equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    logging.info("\n=== BACKTEST RESULTS ===")
    logging.info(f"Strategy Total Return:     {total_return:8.2%}")
    logging.info(f"Buy & Hold Total Return:   {bh_return:8.2%}")
    logging.info(f"Number of trades:          {int(n_trades)}")
    logging.info(f"Win rate:                  {win_rate:8.1%}")
    logging.info(f"Avg return per trade:      {avg_return_per_trade:8.4f}")
    logging.info(f"Sharpe ratio (ann. approx): {sharpe:8.2f}")
    logging.info(f"Max Drawdown:              {max_dd:8.2%}")
    
    # Save results for further analysis/plotting
    output_path = 'data/processed/backtest_results_1h.csv'
    results.to_csv(output_path)
    logging.info(f"Backtest results saved to: {output_path}")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    data = load_data_and_model()
    if data:
        df, proba, prices, returns = data
        run_backtest(df, proba, prices, returns)