import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import subprocess


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

print("Current working directory:", os.getcwd())


def main():
    # ---------- Kaggle download (optional) ----------

    def download_kaggle(dataset, to_folder):
        """
        Tries to download a Kaggle dataset using `python -m kaggle`.
        Returns True if download succeeded, False if anything fails.
        """
        os.makedirs(to_folder, exist_ok=True)
        try:
            import sys
            subprocess.run(
                [
                    sys.executable, "-m", "kaggle", "datasets", "download",
                    "-d", dataset, "-p", to_folder, "--unzip"
                ],
                check=True
            )
            print(f"Kaggle dataset {dataset} downloaded into {to_folder}.")
            return True
        except Exception as e:
            print(f"[WARN] Kaggle download failed for '{dataset}': {e}")
            print("       Will rely on existing CSV files in DATA_DIR/raw instead.")
            return False

    # Try to fetch the two Kaggle datasets, but do not crash if it fails
    download_kaggle("therohk/million-headlines", RAW_DIR)
    download_kaggle("shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025", RAW_DIR)

    # ---------- Functions for data processing ----------

    def flatten_columns(df):
        """Flatten MultiIndex columns if needed."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(filter(None, map(str, col))).strip()
                for col in df.columns.values
            ]
        return df

    def drop_duplicate_dates(df, date_col):
        return df.drop_duplicates(subset=[date_col])

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

    dfs_raw, dfs_proc = {}, {}
    for name, ticker in tickers.items():
        df_raw = download_commodity(ticker, name)
        dfs_raw[name] = df_raw
        if not df_raw.empty:
            dfs_proc[name] = feature_engineer(df_raw, price_cols[name], name)
        else:
            dfs_proc[name] = pd.DataFrame()

    # ---------- GPR loading ----------

    def load_gpr(gpr_path):
        """
        Loads Geopolitical Risk Index "Geopolitical Risk Index Daily.csv"
        from RAW_DIR/All_Historical_Data_Separately (or wherever you stored it),
        resamples to daily frequency, and returns a DataFrame with 'Date' column.
        """
        if not os.path.exists(gpr_path):
            print(f"[WARN] GPR dataset not found at: {gpr_path}")
            return pd.DataFrame()

        gpr = pd.read_csv(gpr_path)
        if "DATE" not in gpr.columns:
            print("[WARN] GPR dataset missing required 'DATE' column")
            return pd.DataFrame()

        gpr["DATE"] = pd.to_datetime(gpr["DATE"], errors="coerce")
        gpr = gpr.dropna(subset=["DATE"]).drop_duplicates(subset=["DATE"])
        gpr = gpr.sort_values("DATE").reset_index(drop=True)

        # Resample to daily frequency with forward fill
        gpr_daily = gpr.set_index("DATE").resample("D").ffill().reset_index()

        keep_cols = [c for c in ["DATE", "GPRD", "GPRD_THREAT", "EVENT"] if c in gpr_daily.columns]
        gpr_daily = gpr_daily[keep_cols]
        gpr_daily = gpr_daily.rename(columns={"DATE": "Date"})

        return gpr_daily

    def merge_with_gpr(df, gpr_df, name):
        if df.empty or gpr_df.empty:
            print(f"Skipping merge for {name}: Empty dataframe(s).")
            return pd.DataFrame()
        df = drop_duplicate_dates(df, "Date")
        merged = pd.merge(df, gpr_df, on="Date", how="left")
        fname = f"{name.lower()}_merged.csv"
        merged.to_csv(os.path.join(MERGED_DIR, fname), index=False)
        print(f"Saved merged data: {fname}")
        return merged

    gpr_path = os.path.join(
        RAW_DIR,
        "All_Historical_Data_Separately",
        "Geopolitical Risk Index Daily.csv"
    )
    gpr_daily = load_gpr(gpr_path)
    dfs_merged = {}
    for name, df_proc in dfs_proc.items():
        dfs_merged[name] = merge_with_gpr(df_proc, gpr_daily, name)

    # ---------- News dataset from abcnews-date-text.csv ----------

    news_path = os.path.join(RAW_DIR, "abcnews-date-text.csv")
    if not os.path.exists(news_path):
        raise FileNotFoundError(
            f"News dataset not found at {news_path}.\n"
            f"Either place abcnews-date-text.csv there manually, "
            f"or ensure Kaggle download is working."
        )

    news_df = pd.read_csv(news_path)
    print("News rows loaded:", len(news_df))

    # abcnews-date-text.csv columns: publish_date (int), headline_text (string) [web:187][web:197]
    news_df["date"] = pd.to_datetime(
        news_df["publish_date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    geo_keywords = [
        "war", "wars", "sanctions", "sanction", "conflict", "conflicts", "geopolitical", "tension", "tensions",
        "embargo", "embargoes", "crisis", "crises", "invasion", "invasions", "terrorism", "opec", "blockade",
        "blockades", "dispute", "disputes", "escalation", "escalations", "hostility", "hostilities", "unrest",
        "strike", "strikes", "alliance", "alliances", "treaty", "treaties", "summit", "summits", "diplomacy",
        "iran", "syria", "syrian", "libya", "lybian", "iraq", "north korea", "ukraine", "russia", "china", "trade war",
        "trade wars",
        "missile", "missiles", "military", "nuclear", "sanctioned", "ceasefire", "ceasefires", "negotiation",
        "negotiations", "occupation", "occupations", "regime", "regimes", "rebel", "rebels", "protest", "protests",
        "cyberattack", "cyberattacks", "espionage", "border", "borders", "refugee", "refugees", "intervention",
        "interventions", "pipeline", "pipelines", "tariff", "tariffs", "boycott", "boycotts", "expulsion",
        "expulsions", "diplomat", "diplomats", "embassy", "embassies", "coalition", "coalitions", "genocide",
        "genocides", "hostage", "hostages", "radical", "radicals", "siege", "sieges", "nato", "chechen", "lebanon",
        "yemen", "taliban", "islamist", "afghanistan", "gaza", "hamas", "kabul", "saddam hussein",
        "global financial crisis", "economic recession", "economic recessions", "chaos", "unemployment", "instability",
        "insolvency", "credit crunch", "unpayable debts", "abkhazia", "ossetia", "separatists", "donetsk", "luhansk",
        "south sudan", "jihadism", "palestine", "palestinian", "isis", "crimea", "annexation", "houthi",
        "migration crises", "donbas", "arab spring", "kosovo", "ukraine war", "ukraine conflict", "AFU"
    ]

    def extract_news_features_abc(news_df, keywords):
        news_df = news_df.copy()
        analyzer = SentimentIntensityAnalyzer()

        news_df["geo_keyword_hits"] = news_df["headline_text"].apply(
            lambda text: sum(kw in text for kw in keywords if isinstance(text, str))
        )

        news_df["sentiment"] = news_df["headline_text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"] if isinstance(x, str) else 0
        )

        news_daily = (
            news_df.groupby("date")
            .agg({"geo_keyword_hits": "sum", "sentiment": "mean"})
            .reset_index()
            .dropna(subset=["date"])
            .sort_values("date")
        )
        return news_daily

    news_features = extract_news_features_abc(news_df, geo_keywords)
    news_features = news_features.rename(columns={"date": "Date"})
    news_features["Date"] = pd.to_datetime(news_features["Date"])

    # ---------- Enrich with news + GPR ----------

    merged_data = {}
    if not gpr_daily.empty:
        gpr_daily["Date"] = pd.to_datetime(gpr_daily["Date"])

    for name, ticker in tickers.items():
        price_col = price_cols[name]
        df_price = dfs_raw.get(name, pd.DataFrame())
        if df_price.empty:
            continue
        df_feat = dfs_proc.get(name, pd.DataFrame())
        if df_feat.empty:
            continue

        df_feat["Date"] = pd.to_datetime(df_feat["Date"])

        if not gpr_daily.empty:
            df_merge = pd.merge_asof(df_feat, gpr_daily, on="Date", direction="backward")
        else:
            df_merge = df_feat.copy()

        df_merge = pd.merge_asof(df_merge, news_features, on="Date", direction="backward")

        if "EVENT" in df_merge.columns:
            df_merge["event_dummy"] = df_merge["EVENT"].notna().astype(int)
        else:
            df_merge["event_dummy"] = 0

        fname = f"{name.lower()}_enriched.csv"
        df_merge.to_csv(os.path.join(ENRICHED_DIR, fname), index=False)
        merged_data[name] = df_merge

    end_date = pd.to_datetime("2021-12-31")
    for name, df in merged_data.items():
        merged_trim = df[df["Date"] <= end_date].reset_index(drop=True)
        fname = f"{name.lower()}_enriched.csv"
        merged_trim.to_csv(os.path.join(ENRICHED_DIR, fname), index=False)

    print("Data collection complete. Enriched datasets saved to", ENRICHED_DIR)


if __name__ == "__main__":
    main()
