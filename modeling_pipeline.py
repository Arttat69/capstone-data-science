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

# Load enriched data
merged_data = {}
for name in commodities:
    fname = f"{name.lower()}_enriched.csv"
    path = os.path.join(ENRICHED_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        merged_data[name] = df
    else:
        print(f"Missing enriched file for {name}")
# Functions for modeling
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
    target_return = 'Return'
    target_ma = 'MA_5'
    lag_features = ['GPRD', 'geo_keyword_hits']

    results = {}
    train_dfs = {}  # To store for LSTM in next cell
    test_dfs = {}  # To store for LSTM in next cell
    features_enhanced_dict = {}  # To store for LSTM

    for commodity, df in merged_data.items():
        print(f"Processing commodity: {commodity}")

        df['geo_keyword_hits'].fillna(0, inplace=True)
        df['sentiment'].fillna(0, inplace=True)
        if 'EVENT' in df.columns:
            df['EVENT'].fillna('None', inplace=True)

        # Add lagged features
        df = add_lagged_features(df, lag_features, max_lag=5)
        df.dropna(inplace=True)
        print(f"{commodity} - {len(df)} rows after lagged features added and NaN drops.")

        features_baseline = ['Return_lag1']
        features_enhanced = ['Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment'] + \
                            [f'{feat}_lag{lag}' for feat in lag_features for lag in range(1, 6)]

        # Convert to numeric and drop any remaining NaNs in features
        df[features_enhanced] = df[features_enhanced].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=features_enhanced + [target_return], inplace=True)

        # Train/test split (adjusted for reasonable data distribution)
        split_date = pd.to_datetime('2018-01-01')
        train_df = df[df['Date'] < split_date].copy()
        test_df = df[df['Date'] >= split_date].copy()

        print(f"{commodity} - Train size: {len(train_df)}, Test size: {len(test_df)}")
        if len(train_df) < 20 or len(test_df) < 20:
            print(f"Insufficient samples for {commodity}, skipping.")
            continue

        # Classical models for Return
        X_train_base, y_train = prepare_features_targets(train_df, features_baseline, target_return)
        X_test_base, y_test = prepare_features_targets(test_df, features_baseline, target_return)
        X_train_enh = train_df[features_enhanced]
        X_test_enh = test_df[features_enhanced]

        scaler = StandardScaler()
        X_train_enh_scaled = scaler.fit_transform(X_train_enh)
        X_test_enh_scaled = scaler.transform(X_test_enh)

        # Baseline LR
        lr_base = LinearRegression()
        lr_base.fit(X_train_base, y_train)
        y_pred_base = lr_base.predict(X_test_base)
        rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

        # Enhanced RF
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_enh_scaled, y_train)
        y_pred_rf = rf.predict(X_test_enh_scaled)
        rmse_enh = np.sqrt(mean_squared_error(y_test, y_pred_rf))

        results[commodity] = {'RMSE_baseline': rmse_base, 'RMSE_enhanced': rmse_enh}

        # Binary classification target
        train_df['Return_binary'] = (train_df['Return'] > 0).astype(int)
        test_df['Return_binary'] = (test_df['Return'] > 0).astype(int)

        # Drop rows with NaNs in classification features
        train_df = train_df.dropna(subset=features_enhanced)
        test_df = test_df.dropna(subset=features_enhanced)

        X_train_class = scaler.fit_transform(train_df[features_enhanced])
        X_test_class = scaler.transform(test_df[features_enhanced])
        y_train_class = train_df['Return_binary']
        y_test_class = test_df['Return_binary']

        # Logistic Regression classifier
        logreg = LogisticRegression(max_iter=200)
        logreg.fit(X_train_class, y_train_class)
        y_pred_class = logreg.predict(X_test_class)
        print(f"{commodity} - Classification Accuracy: {accuracy_score(y_test_class, y_pred_class):.4f}")

        # Clustering for regimes
        kmeans = KMeans(n_clusters=2, random_state=42)
        regime_features = df[['Vol_5', 'GPRD', 'geo_keyword_hits']].fillna(0)
        df['Regime'] = kmeans.fit_predict(regime_features)

        # Store for LSTM in next cell
        train_dfs[commodity] = train_df
        test_dfs[commodity] = test_df
        features_enhanced_dict[commodity] = features_enhanced

        print(f"Finished classical processing for {commodity}\n")

    # Save classical results (move to end if you want)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(MODEL_RESULTS_DIR, 'classical_model_results.csv'))

if __name__ == "__main__":
    main()