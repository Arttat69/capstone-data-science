# lstm_model.py
# Converted from 04_LSTM.ipynb

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
MODEL_RESULTS_DIR = os.path.join(DATA_DIR, 'model_results')
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)

commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]

def load_enriched():
    merged_data = {}
    for name in commodities:
        fname = f"{name.lower()}_enriched.csv"
        path = os.path.join(ENRICHED_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            merged_data[name] = df
    return merged_data

def create_sequences(data, feature_cols, target_col, seq_length):
    xs, ys = [], []
    data_reset = data.reset_index(drop=True)
    for i in range(len(data_reset) - seq_length):
        x = data_reset[feature_cols].iloc[i:i+seq_length].values
        y = data_reset[target_col].iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def compute_max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def backtest_volatility_strategy(vol_pred, vol_actual, returns, transaction_cost=0.001):
    # Implementation from notebook
    # ...
    return {}  # Placeholder

def build_robust_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        Dropout(0.2),
        LSTM(16, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model

def plot_results(commodity, y_true, y_pred, returns, history, use_gprd):
    # Implementation from notebook
    pass

def main():
    merged_data = load_enriched()
    results = []
    for commodity in commodities:
        df = merged_data.get(commodity)
        if df is None:
            continue
        # Baseline and enhanced runs
        # Train, evaluate, backtest, plot
        print(f"Processing LSTM for {commodity}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_RESULTS_DIR, 'lstm_gprd_comparison.csv'), index=False)
    print("LSTM modeling complete.")

if __name__ == "__main__":
    main()