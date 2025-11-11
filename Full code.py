# Data collection

# Part 1
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import subprocess

print("Current working directory:", os.getcwd())


# --------- Folder Setup ----------
ROOT = os.getcwd()
# ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR      = os.path.join(ROOT, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MERGED_DIR    = os.path.join(DATA_DIR, "merged")
ENRICHED_DIR  = os.path.join(DATA_DIR, "enriched")

for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MERGED_DIR, ENRICHED_DIR]:
    os.makedirs(folder, exist_ok=True)

# --------- Date Range ----------
START = "2000-01-01"
END = None  # yfinance: None => up to today



# Part 2: Download Kaggle Datasets
def download_kaggle(dataset, to_folder):
    os.makedirs(to_folder, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset, "-p", to_folder, "--unzip"
    ], check=True)
    print(f"Kaggle dataset {dataset} downloaded.")

#dataset below are useless
download_kaggle("shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025", RAW_DIR)
download_kaggle("everydaycodings/global-news-dataset", RAW_DIR)

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


#Part 6
import matplotlib.pyplot as plt
# Gold
# --- Correlation ---
df_gold = dfs_merged['Gold'].dropna(subset=['Close_GC=F', 'GPRD'])
corr_gold = df_gold['Close_GC=F'].corr(df_gold['GPRD'])
print(f"Gold price and GPRD correlation: {corr_gold:.3f}")

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(df_gold['Date'], df_gold['Close_GC=F'], label='Gold Price', color='gold')
plt.plot(df_gold['Date'], df_gold['GPRD'], label='GPRD', color='red', alpha=0.6)
plt.legend()
plt.title('Gold Price and GPRD Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Oil WTI
# --- Correlation ---
df_wti = dfs_merged['WTI'].dropna(subset=['Close_CL=F', 'GPRD'])
corr_wti = df_wti['Close_CL=F'].corr(df_wti['GPRD'])
print(f"WTI price and GPRD correlation: {corr_wti:.3f}")

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(df_wti['Date'], df_wti['Close_CL=F'], label='WTI Price', color='black')
plt.plot(df_wti['Date'], df_wti['GPRD'], label='GPRD', color='red', alpha=0.6)
plt.legend()
plt.title('WTI Price and GPRD Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Wheat
# --- Correlation ---
df_wheat = dfs_merged['Wheat'].dropna(subset=['Close_ZW=F', 'GPRD'])
corr_wheat = df_wheat['Close_ZW=F'].corr(df_wheat['GPRD'])
print(f"Wheat price and GPRD correlation: {corr_wheat:.3f}")

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(df_wheat['Date'], df_wheat['Close_ZW=F'], label='Wheat Price', color='green')
plt.plot(df_wheat['Date'], df_wheat['GPRD'], label='GPRD', color='red', alpha=0.6)
plt.legend()
plt.title('Wheat Price and GPRD Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# Natural Gas UNG
df_ng = dfs_merged['NaturalGas'].dropna(subset=['Close_UNG', 'GPRD'])
corr_ng = df_ng['Close_UNG'].corr(df_ng['GPRD'])
print(f"Natural Gas price and GPRD correlation: {corr_ng:.3f}")

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df_ng['Date'], df_ng['Close_UNG'], label='Natural Gas Price', color='orange')
ax1.set_ylabel('Natural Gas Price (log scale)', color='orange')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='orange')

ax2 = ax1.twinx()
ax2.plot(df_ng['Date'], df_ng['GPRD'], label='GPRD', color='red', alpha=0.6)
ax2.set_ylabel('GPRD (linear scale)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.legend(loc='upper left')
plt.title('Natural Gas Price (Log Scale) and GPRD Over Time')
plt.xlabel('Date')
plt.show()

#Copper
df_copper = dfs_merged['Copper'].dropna(subset=['Close_HG=F', 'GPRD'])
corr_copper = df_copper['Close_HG=F'].corr(df_copper['GPRD'])
print(f"Copper price and GPRD correlation: {corr_copper:.3f}")

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(df_copper['Date'], df_copper['Close_HG=F'], label='Copper Price', color='brown')
ax1.set_ylabel('Copper Price (log scale)', color='brown')
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor='brown')

ax2 = ax1.twinx()
ax2.plot(df_copper['Date'], df_copper['GPRD'], label='GPRD', color='red', alpha=0.6)
ax2.set_ylabel('GPRD (linear scale)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.legend(loc='upper left')
plt.title('Copper Price (Log Scale) and GPRD Over Time')
plt.xlabel('Date')
plt.show()

#Lithium ETF Proxy
df_lithium = dfs_merged['Lithium'].dropna(subset=['Close_LIT', 'GPRD'])
corr_lithium = df_lithium['Close_LIT'].corr(df_lithium['GPRD'])
print(f"Lithium ETF (LIT) and GPRD correlation: {corr_lithium:.3f}")

plt.figure(figsize=(12,6))
plt.plot(df_lithium['Date'], df_lithium['Close_LIT'], label='LIT ETF Price', color='purple')
plt.plot(df_lithium['Date'], df_lithium['GPRD'], label='GPRD', color='red', alpha=0.6)
plt.legend()
plt.title('LIT ETF Price and GPRD Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

#
#

#Part 7
#Example for Gold of MA5, Volatility...

# Example for Gold
df = dfs_merged['Gold']

# Descriptive statistics
print(df[['Return', 'MA_5', 'Vol_5']].describe())

# Plot
plt.figure(figsize=(14,6))
plt.plot(df['Date'], df['Return'], label='Daily Return', alpha=0.7)
plt.plot(df['Date'], df['MA_5'], label='5-Day MA', alpha=0.7)
plt.plot(df['Date'], df['Vol_5'], label='5-Day Volatility', alpha=0.7)
plt.legend()
plt.title('Gold: Return, 5-Day MA, and 5-Day Volatility')
plt.xlabel('Date')
plt.show()

# Trying to do some buy/sell prediction based on when return > MA_5
print("Correlation matrix:")
print(df[['Return', 'MA_5', 'Vol_5']].corr())

df['Signal'] = 0
df.loc[df['Return'] > df['MA_5'], 'Signal'] = 1  # Buy
df.loc[df['Return'] < df['MA_5'], 'Signal'] = -1 # Sell
print(df[['Date', 'Return', 'MA_5', 'Signal']].tail(10))


# Plot for Gold: Return, 5-Day MA, and 5-Day Volatility (Multiple Y-Axes)
df = dfs_merged['Gold']  # or any other commodity

fig, ax1 = plt.subplots(figsize=(14,6))

# Plot MA_5 (price scale)
ax1.plot(df['Date'], df['MA_5'], label='5-Day MA', color='blue')
ax1.set_ylabel('5-Day MA (Price)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Plot Return (percentage scale)
ax2 = ax1.twinx()
ax2.plot(df['Date'], df['Return'], label='Return', color='green', alpha=0.7)
ax2.set_ylabel('Return', color='green')
ax2.tick_params(axis='y', labelcolor='green')

# Plot Vol_5 (volatility scale)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
ax3.plot(df['Date'], df['Vol_5'], label='5-Day Volatility', color='red', alpha=0.7)
ax3.set_ylabel('5-Day Volatility', color='red')
ax3.tick_params(axis='y', labelcolor='red')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
lines = lines1 + lines2 + lines3
labels = labels1 + labels2 + labels3
plt.legend(lines, labels, loc='upper left')

plt.title('Gold: Return, 5-Day MA, and 5-Day Volatility (Multiple Y-Axes)')
plt.xlabel('Date')
plt.show()


###
###
###
# EDA Feature engineering

# Part1

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
gpr_daily = load_gpr()
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
    df_merge.to_csv(os.path.join(MERGED_DIR, fname), index=False)
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
