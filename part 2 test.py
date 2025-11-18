
# Part 1
#pip install tensorflow

import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from textblob import TextBlob
import subprocess



# Set paths
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MERGED_DIR = os.path.join(DATA_DIR, "merged")
ENRICHED_DIR  = os.path.join(DATA_DIR, "enriched")
MODEL_RESULTS_DIR = os.path.join(DATA_DIR, 'model_results')


for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MERGED_DIR, ENRICHED_DIR, MODEL_RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Commodity tickers, price columns
tickers = {
    "Gold": "GC=F",
    "WTI": "CL=F",
    "Wheat": "ZW=F",
    "NaturalGas":"UNG",
    "Copper": "HG=F",
    "Lithium": "LIT"
}
price_cols = {
    "Gold": "Close_GC=F",
    "WTI": "Close_CL=F",
    "Wheat": "Close_ZW=F",
    "NaturalGas":"Close_UNG",
    "Copper": "Close_HG=F",
    "Lithium": "Close_LIT"
}
START = "2000-01-01"
END = None

print("Current working directory:", os.getcwd())


# Part 2: Download Kaggle Datasets
def download_kaggle(dataset, to_folder):
    os.makedirs(to_folder, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset, "-p", to_folder, "--unzip"
    ], check=True)
    print(f"Kaggle dataset {dataset} downloaded.")


# Part 3
def flatten_columns(df):
    """Flatten MultiIndex columns if needed."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
    return df

def drop_duplicate_dates(df, date_col):
    # Remove rows with duplicated dates
    df = df.drop_duplicates(subset=[date_col])
    return df

def download_commodity(ticker, name):
    print(f"Downloading {name} ({ticker}) ...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True)
    if df.empty:
        print(f"Warning: Empty data for {name}")
        return pd.DataFrame()
    df = flatten_columns(df)
    df.reset_index(inplace=True)
    fname = f"{name.lower()}_raw.csv"
    df.to_csv(os.path.join(RAW_DIR, fname), index=False)
    print(f"Saved raw data: {fname}")
    return df

def feature_engineer(df, price_col, name="Commodity"):
    """
    Feature engineering for commodity price DataFrame.
    Computes return, moving average, volatility, and lagged returns.
    Saves processed file to PROCESSED_DIR if name provided.
    """

    df = df.copy()
    if price_col not in df.columns:
        print(f"{name}: Price column '{price_col}' not found.")
        return pd.DataFrame()

    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df["Return"] = df[price_col].pct_change()
    df["MA_5"] = df[price_col].rolling(5).mean()
    df["Vol_5"] = df["Return"].rolling(5).std()
    df["Return_lag1"] = df["Return"].shift(1)
    df["Return_lag10"] = df["Return"].shift(10)

    # Drop rows with missing required fields
    required_cols = [price_col, "Return", "MA_5", "Vol_5", "Return_lag1", "Return_lag10"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Save processed data if directory and name are provided
    if 'PROCESSED_DIR' in globals():
        fname = f"{name.lower()}_processed.csv"
        save_path = os.path.join(PROCESSED_DIR, fname)
        df.to_csv(save_path, index=False)
        print(f"Saved processed data: {save_path}")

    return df



#Part 4
# ---------- Commodity Tickers & Price Columns ----------
tickers = {
    "Gold":      "GC=F",
    "WTI":       "CL=F",
    "Wheat":     "ZW=F",
    "NaturalGas":"UNG",
    "Copper":    "HG=F",
    "Lithium":   "LIT"    # ETF proxy; no direct lithium spot price available in yfinance
}
price_cols = {
    "Gold":      "Close_GC=F",
    "WTI":       "Close_CL=F",
    "Wheat":     "Close_ZW=F",
    "NaturalGas":"Close_UNG",
    "Copper":    "Close_HG=F",
    "Lithium":   "Close_LIT"
}

# ---------- Download, Process, and Merge ----------
dfs_raw, dfs_proc, dfs_merged = {}, {}, {}

for name, ticker in tickers.items():
    df_raw = download_commodity(ticker, name)
    dfs_raw[name] = df_raw
    if not df_raw.empty:
        dfs_proc[name] = feature_engineer(df_raw, price_cols[name], name)
    else:
        dfs_proc[name] = pd.DataFrame()

# Part 5
def load_gpr(gpr_path):
    """
    Loads and preprocesses daily Geopolitical Risk Index data.
    Resamples to daily frequency with forward fill.
    Returns DataFrame with 'DATE', 'GPRD', 'GPRD_THREAT', and 'EVENT' columns if present.
    """

    if not os.path.exists(gpr_path):
        print(f"GPR dataset not found: {gpr_path}")
        return pd.DataFrame()

    gpr = pd.read_csv(gpr_path)

    if 'DATE' not in gpr.columns:
        print("GPR dataset missing required 'DATE' column")
        return pd.DataFrame()

    gpr['DATE'] = pd.to_datetime(gpr['DATE'], errors='coerce')
    gpr = gpr.dropna(subset=['DATE'])
    gpr = gpr.drop_duplicates(subset=['DATE'])
    gpr = gpr.sort_values('DATE').reset_index(drop=True)

    # Resample to daily frequency and forward-fill missing dates
    gpr_daily = gpr.set_index('DATE').resample('D').ffill().reset_index()

    # Select only relevant columns
    keep_cols = [col for col in ['DATE', 'GPRD', 'GPRD_THREAT', 'EVENT'] if col in gpr_daily.columns]
    gpr_daily = gpr_daily[keep_cols]

    # Rename DATE column to 'Date' for consistency
    gpr_daily = gpr_daily.rename(columns={'DATE':'Date'})

    return gpr_daily


def merge_with_gpr(df, gpr_df, name):
    if df.empty or gpr_df.empty:
        print(f"Skipping merge for {name}: Empty dataframe(s).")
        return pd.DataFrame()
    df = drop_duplicate_dates(df, "Date")
    merged = pd.merge(df, gpr_df, left_on="Date", right_on="Date", how="left")
    fname = f"{name.lower()}_merged.csv"
    merged.to_csv(os.path.join(MERGED_DIR, fname), index=False)
    print(f"Saved merged data: {fname}")
    return merged

# Adapt GPR path as in your structure
gpr_path = os.path.join(RAW_DIR, "All_Historical_Data_Separately", "Geopolitical Risk Index Daily.csv")
gpr_daily = load_gpr(gpr_path)
dfs_merged = {}
for name, df_proc in dfs_proc.items():
    dfs_merged[name] = merge_with_gpr(df_proc, gpr_daily, name)

# ---------- Check Outputs ----------
for name, df_merged in dfs_merged.items():
    if not df_merged.empty:
        print(f"\n{name} merged preview:")
        print(df_merged.head())


import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from textblob import TextBlob

download_kaggle("therohk/million-headlines", RAW_DIR)

news_df = pd.read_csv(os.path.join(RAW_DIR, "abcnews-date-text.csv"))
# Add a datetime 'date' column from yyyymmdd string
news_df['date'] = pd.to_datetime(news_df['publish_date'].astype(str), format='%Y%m%d', errors='coerce')

# Part 2
# Use a robust, expanded geopolitical keywords list
geo_keywords = [
    'war', 'wars', 'sanctions', 'sanction', 'conflict', 'conflicts', 'geopolitical', 'tension', 'tensions',
'embargo', 'embargoes', 'crisis', 'crises', 'invasion', 'invasions', 'terrorism', 'opec', 'blockade',
'blockades', 'dispute', 'disputes', 'escalation', 'escalations', 'hostility', 'hostilities', 'unrest',
'strike', 'strikes', 'alliance', 'alliances', 'treaty', 'treaties', 'summit', 'summits', 'diplomacy',
'iran', 'syria', 'syrian', 'libya', 'lybian', 'iraq', 'north korea', 'ukraine', 'russia', 'china', 'trade war', 'trade wars',
'missile', 'missiles', 'military', 'nuclear', 'sanctioned', 'ceasefire', 'ceasefires', 'negotiation',
'negotiations', 'occupation', 'occupations', 'regime', 'regimes', 'rebel', 'rebels', 'protest', 'protests',
'cyberattack', 'cyberattacks', 'espionage', 'border', 'borders', 'refugee', 'refugees', 'intervention',
'interventions', 'pipeline', 'pipelines', 'tariff', 'tariffs', 'boycott', 'boycotts', 'expulsion',
'expulsions', 'diplomat', 'diplomats', 'embassy', 'embassies', 'coalition', 'coalitions', 'genocide',
'genocides', 'hostage', 'hostages', 'radical', 'radicals', 'siege', 'sieges', 'nato', 'chechen', 'lebanon', 'yemen', 'taliban', 'islamist', 'afghanistan', 'kabul', 'saddam hussein', 'global financial crisis', 'economic recession', 'economic recessions', 'chaos', 'unemployment', 'instability','insolvency', 'credit crunch', 'unpayable debts', 'abkhazia', 'ossetia', 'separatists', 'donetsk', 'luhansk', 'south sudan', 'jihadism', 'palestine', 'palestinian', 'isis', 'crimea', 'annexation', 'houthi', 'migration crises', 'donbas', 'arab spring', 'kosovo', 'ukraine war', 'ukraine conflict', 'AFU'
]

def extract_news_features_abc(news_df, keywords):
    news_df = news_df.copy()
    news_df['geo_keyword_hits'] = news_df['headline_text'].apply(
        lambda text: sum(kw in text for kw in keywords if isinstance(text, str))
    )
    # Sentiment works on string input
    news_df['sentiment'] = news_df['headline_text'].apply(
        lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
    )
    news_daily = news_df.groupby('date').agg({
        'geo_keyword_hits': 'sum',
        'sentiment': 'mean'
    }).reset_index().dropna(subset=['date']).sort_values('date')
    return news_daily

news_features = extract_news_features_abc(news_df, geo_keywords)
print(news_features.head())

# Part 3
# Standardize GPR and News Features BEFORE LOOP (do this ONCE)
gpr_daily = load_gpr(gpr_path)
gpr_daily = gpr_daily.rename(columns={'DATE': 'Date'})  # Rename ONLY IF NEEDED
gpr_daily['Date'] = pd.to_datetime(gpr_daily['Date'])
gpr_daily = gpr_daily.sort_values('Date').reset_index(drop=True)

news_features = news_features.rename(columns={'date': 'Date'})  # Rename ONLY IF NEEDED
news_features['Date'] = pd.to_datetime(news_features['Date'])
news_features = news_features.sort_values('Date').reset_index(drop=True)

merged_data = {}

for name, ticker in tickers.items():
    price_col = price_cols[name]
    df_price = download_commodity(ticker, name)
    if df_price.empty:
        continue
    df_feat = feature_engineer(df_price, price_col)

    # Standardize df_feat 'Date' column
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    df_feat = df_feat.sort_values('Date').reset_index(drop=True)

    # ETL merge part: merge_asof expects sorted 'Date' columns in each DataFrame
    df_merge = pd.merge_asof(df_feat,
                             gpr_daily,
                             on='Date', direction='backward')
    df_merge = pd.merge_asof(df_merge,
                             news_features,
                             on='Date', direction='backward')

    # Event dummy: create if missing
    if 'EVENT' in df_merge.columns:
        df_merge['event_dummy'] = df_merge['EVENT'].notna().astype(int)
    else:
        df_merge['event_dummy'] = 0

    # Save output
    fname = f"{name.lower()}_enriched.csv"
    df_merge.to_csv(os.path.join(ENRICHED_DIR, fname), index=False)
    merged_data[name] = df_merge

# Part 4
for name, df in merged_data.items():
    print(f"\n{name} Correlation Matrix:")
    print(df[['Return', 'MA_5', 'Vol_5', 'GPRD', 'geo_keyword_hits', 'sentiment', "Return_lag10"]].corr())

end_date = pd.to_datetime("2021-12-31")

for name, df in merged_data.items():
    # Trim dataframes to geo news coverage period
    merged_data[name] = df[df['Date'] <= end_date].reset_index(drop=True)


# Part 5 EDA
# --- EDA: Correlations, Stationarity, Distributions ---
# add return lag10,
for name, df in merged_data.items():
    print(f"\n--- EDA for {name} ---")
    print(df[['Return','MA_5','Vol_5','Return_lag1','GPRD','geo_keyword_hits','sentiment', "Return_lag10"]].describe())
    # Correlations
    print("Correlation matrix:\n", df[['Return','MA_5','Vol_5','Return_lag1','GPRD','geo_keyword_hits','sentiment', "Return_lag10"]].corr())
    # Distribution plots
    df[['Return','MA_5','Vol_5','GPRD','geo_keyword_hits']].hist(figsize=(10,8), bins=50)
    plt.suptitle(f'{name} Feature Distributions')
    plt.show()
    # Stationarity test
    print("ADF Test for Price:")
    adf_result = adfuller(df[price_cols[name]])
    print(f"ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3e}")
    print("ADF Test for Return:")
    adf_result = adfuller(df['Return'])
    print(f"ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3e}")
    # Correlation heatmap
    corr = df[['Return','MA_5','Vol_5','Return_lag1','GPRD','geo_keyword_hits','sentiment', "Return_lag10"]].corr()
    sns.heatmap(corr, annot=True, cmap='YlGnBu')
    plt.title(f'{name} Feature Correlation Heatmap')
    plt.show()
    # Time series multi-axis plot for enriched features (example)
    fig, ax1 = plt.subplots(figsize=(14,7))
    ax1.plot(df['Date'], df[price_cols[name]], label='Price', color='blue', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Return'], label='Return', color='green', alpha=0.5)
    ax2.plot(df['Date'], df['Vol_5'], label='Vol_5', color='red', alpha=0.5)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(df['Date'], df['geo_keyword_hits'], label='Geo News Hits', color='purple', alpha=0.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Return / Volatility')
    ax3.set_ylabel('Geo News Hits')
    ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    ax3.legend(loc='upper right')
    plt.title(f'{name}: Price, Return, Volatility, Geo News Hits')
    plt.show()

# Part 6: # Stationarity Tests with Augmented Dickey-Fuller (ADF)

from statsmodels.tsa.stattools import adfuller

for name, df in merged_data.items():
    print(f"\n--- ADF Stationarity Tests for {name} ---")
    # Test price series
    adf_price = adfuller(df[price_cols[name]].dropna())
    print(f"Price: ADF Statistic={adf_price[0]:.3f}, p-value={adf_price[1]:.3g}")
    # Test returns series
    adf_return = adfuller(df['Return'].dropna())
    print(f"Return: ADF Statistic={adf_return[0]:.3f}, p-value={adf_return[1]:.3g}")

    # INTERPRET
    if adf_price[1]<0.05:
        print("Price: Stationary (reject H0)")
    else:
        print("Price: Not stationary (fail to reject H0)")
    if adf_return[1]<0.05:
        print("Return: Stationary (reject H0)")
    else:
        print("Return: Not stationary (fail to reject H0)")


for name, df in merged_data.items():
    print(f"\n--- Stationarity (ADF) for {name} ---")
    # MA5 stationarity (strong correlation with geo_keywords)
    if 'MA_5' in df.columns:
        adf_ma5 = adfuller(df['MA_5'].dropna())
        print(f"MA5:  ADF Statistic={adf_ma5[0]:.3f}, p-value={adf_ma5[1]:.3g} -- ", end="")
        print("Stationary" if adf_ma5[1]<0.05 else "Not stationary")
    # Volatility stationarity
    if 'Vol_5' in df.columns:
        adf_vol = adfuller(df['Vol_5'].dropna())
        print(f"Vol_5: ADF Statistic={adf_vol[0]:.3f}, p-value={adf_vol[1]:.3g} -- ", end="")
        print("Stationary" if adf_vol[1]<0.05 else "Not stationary")


# Part 7 : # Event Analysis: Pre/Post-Event Comparison and Plotting

event_window = 10  # Days before and after event

for name, df in merged_data.items():
    print(f"\n--- Event Analysis for {name} ---")
    if 'event_dummy' in df.columns and df['event_dummy'].sum() > 0:
        event_indices = df.index[df['event_dummy']==1].tolist()
        pre_event_returns, post_event_returns = [], []
        for idx in event_indices:
            if idx>event_window and idx<len(df)-event_window:
                pre_event_returns.append(df.loc[idx-event_window:idx-1, 'Return'].mean())
                post_event_returns.append(df.loc[idx+1:idx+event_window, 'Return'].mean())

        print("Average pre-event return:", np.mean(pre_event_returns))
        print("Average post-event return:", np.mean(post_event_returns))

        # Plot for one example event window (the first event)
        if event_indices:
            idx = event_indices[0]
            fig, ax = plt.subplots(figsize=(10,5))
            window_df = df.loc[idx-event_window:idx+event_window]
            window_df = window_df.set_index('Date')
            window_df['Return'].plot(ax=ax, marker='o')
            ax.axvline(df.loc[idx, 'Date'], color='red', linestyle='--', label='Event')
            plt.title(f"{name} - Return Around Event")
            plt.legend()
            plt.show()
    else:
        print("No event dummies found in data, skipping event analysis.")

# Part 8: Event Analysis with mean/volatility pre/post event

for name, df in merged_data.items():
    print(f"\n--- Event Analysis and Visualization for {name} ---")
    # Determine event dates: 2008 for most, July 2020 for lithium
    if name.lower() == 'lithium':
        event_dates = df[(df['Date'] >= '2020-07-01') & (df['Date'] < '2020-08-01')]['Date']
    else:
        event_dates = df[df['Date'].dt.year == 2008]['Date']

    # Plot MA_5 with all events marked
    if 'MA_5' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['MA_5'], label='MA_5')
        for ed in event_dates:
            plt.axvline(ed, color='red', linestyle='--', alpha=0.4)
        plt.title(f"{name} - MA_5 with Event Markers")
        plt.xlabel("Date")
        plt.ylabel("MA_5")
        plt.legend()
        plt.show()

    # Plot Vol_5 with all events marked
    if 'Vol_5' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Vol_5'], color='orange', label='Vol_5')
        for ed in event_dates:
            plt.axvline(ed, color='red', linestyle='--', alpha=0.4)
        plt.title(f"{name} - Vol_5 with Event Markers")
        plt.xlabel("Date")
        plt.ylabel("Vol_5")
        plt.legend()
        plt.show()

    # Optional: Print mean volatility pre/post-event for quick check
    if len(event_dates) > 0 and 'Vol_5' in df.columns:
        # Use first event only for simple analysis
        idx = df.index[df['Date']==event_dates.iloc[0]][0]
        window = 10
        if idx > window and idx < len(df)-window:
            pre = df.loc[idx-window:idx-1, 'Vol_5'].mean()
            post = df.loc[idx+1:idx+window, 'Vol_5'].mean()
            print(f"Mean volatility pre-event Vol_5:  {pre:.4f}")
            print(f"Mean volatility post-event Vol_5: {post:.4f}")


# Part 3
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score , mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout, Bidirectional

# Assuming functions and variables from your file are defined:
# load_gpr(), news_features (extracted/preprocessed), tickers, pricecols, ENRICHEDDIR, download_commodity(), feature_engineer()


# Store DataFrame in merged_data dictionary for modeling

def prepare_features_targets(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

# Demonstration: modeling pipeline for 'Gold' commodity as example
commodity = 'Gold'
df = merged_data[commodity]

# Define baseline and enhanced feature sets
features_baseline = ['Return_lag1']
features_enhanced = ['Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment']
# We don't include event_dummy since it doesn't have a lot of values which will create trouble when we drop NaN values

target = 'Return'

# Split data into train and test sets by date (time series split, no shuffling)
split_date = pd.to_datetime('2000-01-01')
for commodity, df in merged_data.items():
    print(f"Processing commodity: {commodity}")

    # Convert enhanced features to numeric forcing NaN on errors
    df[features_enhanced] = df[features_enhanced].apply(pd.to_numeric, errors='coerce')

    # Time-series split into train and test
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    # Prepare baseline features
    X_train_base, y_train = prepare_features_targets(train_df, features_baseline, target)
    X_test_base, y_test = prepare_features_targets(test_df, features_baseline, target)

    # Prepare enhanced features
    X_train_enh, _ = prepare_features_targets(train_df, features_enhanced, target)
    X_test_enh, _ = prepare_features_targets(test_df, features_enhanced, target)
    print(f"{commodity} - Train DF shape before scaling: {train_df.shape}")
    print(f"{commodity} - Train features shape before scaling: {X_train_enh.shape}")
    print(f"{commodity} - NaNs in training features:\n{X_train_enh.isna().sum()}")
    if X_train_enh.shape[0] == 0:
        print(f"No training samples for enhanced features for {commodity}, skipping.")
        continue

    # Scale enhanced features
    scaler = StandardScaler()
    X_train_enh_scaled = scaler.fit_transform(X_train_enh)
    X_test_enh_scaled = scaler.transform(X_test_enh)

    # Baseline Linear Regression
    lr_base = LinearRegression()
    lr_base.fit(X_train_base, y_train)
    y_pred_base = lr_base.predict(X_test_base)
    print(f"{commodity} - Baseline model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}")

    # Enhanced Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_enh_scaled, y_train)
    y_pred_rf = rf.predict(X_test_enh_scaled)
    print(f"{commodity} - Enhanced model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")

    # Binary classification target
    train_df.loc[:, 'Return_binary'] = (train_df['Return'] > 0).astype(int)
    test_df.loc[:, 'Return_binary'] = (test_df['Return'] > 0).astype(int)

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

    print(f"Finished processing {commodity}\n")
######


# LSTM sequence forecasting setup
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

# Add lag features to capture delayed geopolitical impact
def add_lagged_features(df, feature_cols, max_lag=5):
    for col in feature_cols:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df
# Set data directory and ensure model results folder exists
DATA_DIR = os.path.abspath("data")  # set your data folder path accordingly
model_results_dir = os.path.join(DATA_DIR, 'model_results')
os.makedirs(model_results_dir, exist_ok=True)

# Prepare target columns and lag features
target_return = 'Return'
target_ma = 'MA_5'  # assume MA_5 column exists in merged_data dfs representing 5-day moving average returns
lag_features = ['GPRD', 'geo_keyword_hits']

results = {}
results_ma = {}

model_results_dir = os.path.join(DATA_DIR, 'model_results')
os.makedirs(model_results_dir, exist_ok=True)

for commodity, df in merged_data.items():
    print(f"Processing commodity: {commodity}")

    df['geo_keyword_hits'].fillna(0, inplace=True)
    df['sentiment'].fillna(0, inplace=True)
    df['EVENT'].fillna('None', inplace=True)

    # Add lagged geopolitical features
    df = add_lagged_features(df, lag_features, max_lag=5)
    print(f"Date range after lag drop: {df['Date'].min()} to {df['Date'].max()}")

    # Drop NA created by lagging
    df.dropna(inplace=True)
    print(f"{commodity} - {len(df)} rows after lagged features added and NaN drops.")

    features_baseline = ['Return_lag1']
    # Use enhanced features including lagged versions
    features_enhanced = ['Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment'] + \
                        [f'{feat}_lag{lag}' for feat in lag_features for lag in range(1, 6)]

    # Convert features to numeric
    df[features_enhanced] = df[features_enhanced].apply(pd.to_numeric, errors='coerce')

    # Train/test split by date
    split_date = pd.to_datetime('2000-01-01')
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()

    print(f"{commodity} - Train size after split: {len(train_df)}, Test size: {len(test_df)}")
    if len(train_df) < 20 or len(test_df) < 20:
        print(f"Insufficient train or test samples for {commodity}, skipping this commodity.")
        continue

    # Prepare baseline and enhanced features for 'Return' target
    X_train_base, y_train = train_df[features_baseline], train_df[target_return]
    X_test_base, y_test = test_df[features_baseline], test_df[target_return]
    X_train_enh, _ = train_df[features_enhanced], train_df[target_return]
    X_test_enh, _ = test_df[features_enhanced], test_df[target_return]

    # Scale enhanced features for models that require it
    scaler = StandardScaler()
    X_train_enh_scaled = scaler.fit_transform(X_train_enh)
    X_test_enh_scaled = scaler.transform(X_test_enh)

    # Baseline Linear Regression for return
    lr_base = LinearRegression()
    lr_base.fit(X_train_base, y_train)
    y_pred_base = lr_base.predict(X_test_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

    # Enhanced Random Forest for return
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_enh_scaled, y_train)
    y_pred_rf = rf.predict(X_test_enh_scaled)
    rmse_enh = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    results[commodity] = {'RMSE_baseline': rmse_base, 'RMSE_enhanced': rmse_enh}

    ## Repeat above for MA_5 target ##
    # Prepare features and target for MA_5
    y_train_ma = train_df[target_ma]
    y_test_ma = test_df[target_ma]

    # Scale features again (MinMaxScaler for LSTM)
    minmax_scaler = MinMaxScaler()
    train_features_scaled = minmax_scaler.fit_transform(X_train_enh)
    test_features_scaled = minmax_scaler.transform(X_test_enh)

    train_scaled_df = pd.DataFrame(train_features_scaled, columns=features_enhanced, index=train_df.index)
    test_scaled_df = pd.DataFrame(test_features_scaled, columns=features_enhanced, index=test_df.index)

    train_scaled_df[target_ma] = y_train_ma
    test_scaled_df[target_ma] = y_test_ma

    # Drop NaNs
    train_scaled_df.dropna(subset=features_enhanced + [target_ma], inplace=True)
    test_scaled_df.dropna(subset=features_enhanced + [target_ma], inplace=True)

    seq_length = 10
    X_train_seq, y_train_seq = create_sequences(train_scaled_df, features_enhanced, target_ma, seq_length)
    X_test_seq, y_test_seq = create_sequences(test_scaled_df, features_enhanced, target_ma, seq_length)
    print(f"{commodity} - Training sequences created: {len(X_train_seq)}, Testing sequences: {len(X_test_seq)}")

    print(f"{commodity} - Training sequences: {len(X_train_seq)}, Testing sequences: {len(X_test_seq)}")
    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
        print(f"Skipping {commodity} due to insufficient sequence samples for LSTM.")
        continue
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Define LSTM model
    model = Sequential([
        Input(shape=(seq_length, len(features_enhanced))),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Adapt validation split dynamically
    val_split = 0.2 if len(X_train_seq) > 10 else 0.0
    callbacks_list = [early_stop] if val_split > 0 else []

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )

    y_pred_lstm = model.predict(X_test_seq)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))
    mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
    mape_lstm = mean_absolute_percentage_error(y_test_seq, y_pred_lstm)
    r2_lstm = r2_score(y_test_seq, y_pred_lstm)

    results_ma[commodity] = {'RMSE_LSTM': rmse_lstm, 'MAE_LSTM': mae_lstm, 'MAPE_LSTM': mape_lstm, 'R2_LSTM': r2_lstm}
    print(f"Finished processing {commodity}\n")

# Save results to csv
results_df = pd.DataFrame(results).T
results_ma_df = pd.DataFrame(results_ma).T
results_df.to_csv(os.path.join(model_results_dir, 'classical_model_results.csv'))
results_ma_df.to_csv(os.path.join(model_results_dir, 'lstm_ma_results.csv'))

print(f"Saved classical and LSTM results to {model_results_dir}")

#Due to very low R² and LSTM prediction based on return, we add also MA_5 as the target#

