import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# Load and preprocess GPR data as before...
gpr_path = os.path.join(RAW_DIR, "AllHistoricalDataSeparately", "Geopolitical Risk Index Daily.csv")
gpr_daily = load_gpr(gpr_path)
gpr_daily = gpr_daily.rename(columns={'DATE': 'Date'})
gpr_daily['Date'] = pd.to_datetime(gpr_daily['Date'])
gpr_daily = gpr_daily.sort_values('Date').reset_index(drop=True)

news_features = news_features.rename(columns={'date': 'Date'})
news_features['Date'] = pd.to_datetime(news_features['Date'])
news_features = news_features.sort_values('Date').reset_index(drop=True)

merged_data = {}

for name, ticker in tickers.items():
    price_col = pricecols[name]
    df_price = download_commodity(ticker, name)
    if df_price.empty:
        continue

    df_feat = feature_engineer(df_price, price_col, name)
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    df_feat = df_feat.sort_values('Date').reset_index(drop=True)

    df_merge = pd.merge_asof(df_feat, gpr_daily, on='Date', direction='backward')
    df_merge = pd.merge_asof(df_merge, news_features, on='Date', direction='backward')

    if 'EVENT' in df_merge.columns:
        df_merge['event_dummy'] = df_merge['EVENT'].notna().astype(int)
    else:
        df_merge['event_dummy'] = 0

    fname = f"{name.lower()}_enriched.csv"
    df_merge.to_csv(os.path.join(ENRICHEDDIR, fname), index=False)

    merged_data[name] = df_merge


def prepare_features_targets(df, features, target):
    X = df[features]
    y = df[target]
    return X, y


commodity = 'Gold'
df = merged_data[commodity]

# Convert feature columns to numeric to ensure compatibility with sklearn and keras
features_baseline = ['Return_lag1']
features_enhanced = ['Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment', 'event_dummy']

# Force numeric conversion with coerce to NaN on errors
df[features_enhanced] = df[features_enhanced].apply(pd.to_numeric, errors='coerce')

target = 'Return'

split_date = pd.to_datetime('2019-01-01')
# Make explicit copies to avoid SettingWithCopyWarning
train_df = df[df['Date'] < split_date].copy()
test_df = df[df['Date'] >= split_date].copy()

X_train_base, y_train = prepare_features_targets(train_df, features_baseline, target)
X_test_base, y_test = prepare_features_targets(test_df, features_baseline, target)

X_train_enh, _ = prepare_features_targets(train_df, features_enhanced, target)
X_test_enh, _ = prepare_features_targets(test_df, features_enhanced, target)

scaler = StandardScaler()
X_train_enh_scaled = scaler.fit_transform(X_train_enh)
X_test_enh_scaled = scaler.transform(X_test_enh)

lr_base = LinearRegression()
lr_base.fit(X_train_base, y_train)
y_pred_base = lr_base.predict(X_test_base)
print(f"Baseline model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_enh_scaled, y_train)
y_pred_rf = rf.predict(X_test_enh_scaled)
print(f"Enhanced model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")

# Create classification binary target explicitly with .loc to avoid warning
train_df.loc[:, 'Return_binary'] = (train_df['Return'] > 0).astype(int)
test_df.loc[:, 'Return_binary'] = (test_df['Return'] > 0).astype(int)

# Drop any rows with NaN in features_enhanced before classification
train_df = train_df.dropna(subset=features_enhanced)
test_df = test_df.dropna(subset=features_enhanced)

X_train_class = scaler.fit_transform(train_df[features_enhanced])
X_test_class = scaler.transform(test_df[features_enhanced])
y_train_class = train_df['Return_binary']
y_test_class = test_df['Return_binary']

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train_class, y_train_class)
y_pred_class = logreg.predict(X_test_class)
print(f"Classification Accuracy: {accuracy_score(y_test_class, y_pred_class):.4f}")

kmeans = KMeans(n_clusters=2, random_state=42)
regime_features = df[['Vol_5', 'GPRD', 'geo_keyword_hits']].fillna(0)
df['Regime'] = kmeans.fit_predict(regime_features)

def create_sequences(data, feature_cols, target_col, seq_length=10):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[feature_cols].iloc[i:(i + seq_length)].values
        y = data[target_col].iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
feature_cols = features_enhanced
target_col = target

# Ensure no NaNs in features before creating sequences
df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

X_seq, y_seq = create_sequences(df, feature_cols, target_col, seq_length)

split_idx = len(train_df) - seq_length
X_train_seq, y_train_seq = X_seq[:split_idx], y_seq[:split_idx]
X_test_seq, y_test_seq = X_seq[split_idx:], y_seq[split_idx:]

# Build LSTM model with explicit Input layer to avoid Keras warning
model = Sequential([
    Input(shape=(seq_length, len(feature_cols))),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=2)

y_pred_lstm = model.predict(X_test_seq)
print(f"LSTM Test RMSE: {np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm)):.4f}"


######
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

def train_and_evaluate_lstm(df, features, target, seq_length=30):
    # Split train/test by date and copy to avoid SettingWithCopy
    split_date = pd.to_datetime('2019-01-01')
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    # MinMax scaling fit on train only and transform both
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_df[features])
    test_features = scaler.transform(test_df[features])

    train_scaled = pd.DataFrame(train_features, columns=features, index=train_df.index)
    test_scaled = pd.DataFrame(test_features, columns=features, index=test_df.index)

    train_scaled[target] = train_df[target]
    test_scaled[target] = test_df[target]

    # Drop NaN in features or target for robust sequence creation
    train_scaled.dropna(subset=features + [target], inplace=True)
    test_scaled.dropna(subset=features + [target], inplace=True)

    X_train, y_train = create_sequences(train_scaled, features, target, seq_length)
    X_test, y_test = create_sequences(test_scaled, features, target, seq_length)

    # Build LSTM model with BiLSTM layers and dropout
    model = Sequential([
        Input(shape=(seq_length, len(features))),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Results for target {target}: RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.2f}%, R2 = {r2:.5f}")

    return model, history, (rmse, mae, mape, r2)

# Example: apply to all commodities in merged_data dict
features_enhanced = ['Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment', 'event_dummy']

results = {}
for commodity, df in merged_data.items():
    print(f"\nTraining LSTM for {commodity}")
    model, history, metrics = train_and_evaluate_lstm(df, features_enhanced, 'Return', seq_length=30)
    results[commodity] = metrics

# Optionally, tabulate or save results for reporting

# Assuming results dict is {commodity: (rmse, mae, mape, r2)}
# Convert to DataFrame for easy saving and reporting
results_df = pd.DataFrame.from_dict(
    results,
    orient='index',
    columns=['RMSE', 'MAE', 'MAPE', 'R2']
)

# Save results to CSV
results_csv_path = os.path.join(MODEL_RESULTS_DIR, 'lstm_model_performance_summary.csv')
results_df.to_csv(results_csv_path)

print(f"LSTM model results saved to {results_csv_path}")
