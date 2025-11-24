# modeling_pipeline.py
# Converted from 03_Modeling Pipeline Setup.ipynb

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
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

def prepare_features_targets(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_sequences(data, feature_cols, target_col, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[feature_cols].iloc[i:i+seq_length].values
        y = data[target_col].iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def add_lagged_features(df, feature_cols, max_lag=5):
    for col in feature_cols:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def main():
    merged_data = load_enriched()
    # Implement classical models, LSTM, etc.
    # For example:
    results_lstm = {}
    for commodity, df in merged_data.items():
        # Placeholder for modeling code
        # Add lagged features, train/test split, models, etc.
        print(f"Processing modeling for {commodity}")
        # Save results

    print("Modeling pipeline complete.")

if __name__ == "__main__":
    main()