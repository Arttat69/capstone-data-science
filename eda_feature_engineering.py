import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Set paths (same as notebook 1)
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")

# Commodity names (from tickers)
commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]
price_cols = {
    "Gold": "Close_GC=F",
    "WTI": "Close_CL=F",
    "Wheat": "Close_ZW=F",
    "NaturalGas": "Close_UNG",
    "Copper": "Close_HG=F",
    "Lithium": "Close_LIT"
}

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

def main():
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
        # Time series multi-axis plot
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

    # Additional Stationarity Tests
    for name, df in merged_data.items():
        print(f"\n--- ADF Stationarity Tests for {name} ---")
        # Test price series
        adf_price = adfuller(df[price_cols[name]].dropna())
        print(f"Price: ADF Statistic={adf_price[0]:.3f}, p-value={adf_price[1]:.3g}")
        # Test returns series
        adf_return = adfuller(df['Return'].dropna())
        print(f"Return: ADF Statistic={adf_return[0]:.3f}, p-value={adf_return[1]:.3g}")

        # Interpret
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
        # MA5 stationarity
        if 'MA_5' in df.columns:
            adf_ma5 = adfuller(df['MA_5'].dropna())
            print(f"MA5:  ADF Statistic={adf_ma5[0]:.3f}, p-value={adf_ma5[1]:.3g} -- ", end="")
            print("Stationary" if adf_ma5[1]<0.05 else "Not stationary")
        # Volatility stationarity
        if 'Vol_5' in df.columns:
            adf_vol = adfuller(df['Vol_5'].dropna())
            print(f"Vol_5: ADF Statistic={adf_vol[0]:.3f}, p-value={adf_vol[1]:.3g} -- ", end="")
            print("Stationary" if adf_vol[1]<0.05 else "Not stationary")

    # Event Analysis
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

        print("EDA complete.")


if __name__ == "__main__":
    # This allows you to run this file individually for testing
    main()