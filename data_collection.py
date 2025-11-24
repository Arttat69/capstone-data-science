import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import subprocess
from textblob import TextBlob

# Set paths
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MERGED_DIR = os.path.join(DATA_DIR, "merged")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")

for folder in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MERGED_DIR, ENRICHED_DIR]:
    os.makedirs(folder, exist_ok=True)

# Commodity tickers and price columns
tickers = {
    "Gold": "GC=F",
    "WTI": "CL=F",
    "Wheat": "ZW=F",
    "NaturalGas": "UNG",
    "Copper": "HG=F",
    "Lithium": "LIT"
}
price_cols = {
    "Gold": "Close_GC=F",
    "WTI": "Close_CL=F",
    "Wheat": "Close_ZW=F",
    "NaturalGas": "Close_UNG",
    "Copper": "Close_HG=F",
    "Lithium": "Close_LIT"
}
START = "2000-01-01"
END = None

def download_kaggle(dataset, to_folder):
    os.makedirs(to_folder, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset, "-p", to_folder, "--unzip"
    ], check=True)
    print(f"Kaggle dataset {dataset} downloaded.")

# Functions for data processing

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(filter(None, map(str, col))).strip() for col in df.columns.values]
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

    required_cols = [price_col, "Return", "MA_5", "Vol_5", "Return_lag1", "Return_lag10"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    fname = f"{name.lower()}_processed.csv"
    save_path = os.path.join(PROCESSED_DIR, fname)
    df.to_csv(save_path, index=False)
    print(f"Saved processed data: {save_path}")

    return df

def main():
    # Download Kaggle datasets (uncomment if needed)
    # download_kaggle("therohk/million-headlines", RAW_DIR)
    # download_kaggle("shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025", RAW_DIR)

    dfs_raw, dfs_proc = {}, {}
    for name, ticker in tickers.items():
        df_raw = download_commodity(ticker, name)
        dfs_raw[name] = df_raw
        if not df_raw.empty:
            dfs_proc[name] = feature_engineer(df_raw, "Close", name)  # Adjusted to "Close" as per yf.download

    # Load GPR (adjust path if needed)
    gpr_path = os.path.join(RAW_DIR, "GPRD_Historical.csv")
    if os.path.exists(gpr_path):
        gpr_daily = pd.read_csv(gpr_path)
        gpr_daily['Date'] = pd.to_datetime(gpr_daily['Date'])
        gpr_daily = gpr_daily.sort_values('Date').reset_index(drop=True)
        gpr_daily = gpr_daily.set_index('Date').resample('D').ffill().reset_index()
    else:
        print("GPR file not found.")
        return

    # Load news
    news_path = os.path.join(RAW_DIR, "abcnews-date-text.csv")
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path)
        news_df['date'] = pd.to_datetime(news_df['publish_date'].astype(str), format='%Y%m%d', errors='coerce')
    else:
        print("News file not found.")
        return

    geo_keywords = [
        'war', 'wars', 'sanctions', 'sanction', 'conflict', 'conflicts', 'geopolitical', 'tension', 'tensions',
        'embargo', 'embargoes', 'crisis', 'crises', 'invasion', 'invasions', 'terrorism', 'opec', 'blockade',
        'blockades', 'dispute', 'disputes', 'escalation', 'escalations', 'hostility', 'hostilities', 'unrest',
        'strike', 'strikes', 'alliance', 'alliances', 'treaty', 'treaties', 'summit', 'summits', 'diplomacy',
        'iran', 'syria', 'syrian', 'libya', 'lybian', 'iraq', 'north korea', 'ukraine', 'russia', 'china', 'trade war',
        'trade wars',
        'missile', 'missiles', 'military', 'nuclear', 'sanctioned', 'ceasefire', 'ceasefires', 'negotiation',
        'negotiations', 'occupation', 'occupations', 'regime', 'regimes', 'rebel', 'rebels', 'protest', 'protests',
        'cyberattack', 'cyberattacks', 'espionage', 'border', 'borders', 'refugee', 'refugees', 'intervention',
        'interventions', 'pipeline', 'pipelines', 'tariff', 'tariffs', 'boycott', 'boycotts', 'expulsion',
        'expulsions', 'diplomat', 'diplomats', 'embassy', 'embassies', 'coalition', 'coalitions', 'genocide',
        'genocides', 'hostage', 'hostages', 'radical', 'radicals', 'siege', 'sieges', 'nato', 'chechen', 'lebanon',
        'yemen', 'taliban', 'islamist', 'afghanistan', 'gaza', 'hamas', 'kabul', 'saddam hussein',
        'global financial crisis', 'economic recession', 'economic recessions', 'chaos', 'unemployment', 'instability',
        'insolvency', 'credit crunch', 'unpayable debts', 'abkhazia', 'ossetia', 'separatists', 'donetsk', 'luhansk',
        'south sudan', 'jihadism', 'palestine', 'palestinian', 'isis', 'crimea', 'annexation', 'houthi',
        'migration crises', 'donbas', 'arab spring', 'kosovo', 'ukraine war', 'ukraine conflict', 'AFU'
    ]

    def extract_news_features_abc(news_df, keywords):
        news_df = news_df.copy()
        news_df['geo_keyword_hits'] = news_df['headline_text'].apply(
            lambda text: sum(kw in text.lower() for kw in keywords if isinstance(text, str))
        )
        news_df['sentiment'] = news_df['headline_text'].apply(
            lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0
        )
        news_daily = news_df.groupby('date').agg({
            'geo_keyword_hits': 'sum',
            'sentiment': 'mean'
        }).reset_index().dropna(subset=['date']).sort_values('date')
        return news_daily

    news_features = extract_news_features_abc(news_df, geo_keywords)
    news_features = news_features.rename(columns={'date': 'Date'})
    news_features['Date'] = pd.to_datetime(news_features['Date'])

    # Enrich with news features
    merged_data = {}
    gpr_daily['Date'] = pd.to_datetime(gpr_daily['Date'])

    for name, ticker in tickers.items():
        price_col = price_cols[name]
        df_price = dfs_raw.get(name, pd.DataFrame())  # Use raw or processed? Using processed for features
        if df_price.empty:
            continue
        df_feat = dfs_proc.get(name, pd.DataFrame())

        df_feat['Date'] = pd.to_datetime(df_feat['Date'])

        # Merge with GPR and news using merge_asof
        df_merge = pd.merge_asof(df_feat, gpr_daily, on='Date', direction='backward')
        df_merge = pd.merge_asof(df_merge, news_features, on='Date', direction='backward')

        # Event dummy
        if 'EVENT' in df_merge.columns:
            df_merge['event_dummy'] = df_merge['EVENT'].notna().astype(int)
        else:
            df_merge['event_dummy'] = 0

        # Save enriched
        fname = f"{name.lower()}_enriched.csv"
        df_merge.to_csv(os.path.join(ENRICHED_DIR, fname), index=False)
        merged_data[name] = df_merge

    # Trim if needed
    end_date = pd.to_datetime("2024-12-31")
    for name, df in merged_data.items():
        df = df[df['Date'] <= end_date].reset_index(drop=True)
        fname = f"{name.lower()}_enriched.csv"
        df.to_csv(os.path.join(ENRICHED_DIR, fname), index=False)

    print("Data collection complete.")

if __name__ == "__main__":
    main()