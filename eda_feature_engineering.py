"""
Publication-Quality Appendix Figure Generator
Generates 4 key figures for academic report appendices
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100
sns.set_style("whitegrid")

# Set paths
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
APPENDIX_DIR = os.path.join(OUTPUT_DIR, "appendix_plots")

# Create output directory
os.makedirs(APPENDIX_DIR, exist_ok=True)

# Commodity configuration
commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]
price_cols = {
    "Gold": "Close_GC=F",
    "WTI": "Close_CL=F",
    "Wheat": "Close_ZW=F",
    "NaturalGas": "Close_UNG",
    "Copper": "Close_HG=F",
    "Lithium": "Close_LIT"
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_enriched_data():
    """Load all enriched commodity data files."""
    merged_data = {}
    print("📂 Loading enriched data...")

    for name in commodities:
        fname = f"{name.lower()}_enriched.csv"
        path = os.path.join(ENRICHED_DIR, fname)

        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            merged_data[name] = df
            print(f"  ✓ {name}: {len(df)} rows")
        else:
            print(f"  ✗ Missing: {fname}")

    print(f"\n✅ Loaded {len(merged_data)} commodities\n")
    return merged_data


# =============================================================================
# APPENDIX FIGURE 1: CORRELATION HEATMAPS
# =============================================================================

def plot_correlation_heatmaps(merged_data):
    """
    Create 6 correlation heatmaps (one per commodity) in a single figure.
    Highlights multicollinearity between GPRD and geo_keyword_hits.
    """
    print("📊 Generating Appendix F: Correlation Heatmaps...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Feature Correlation Matrices by Commodity',
                 fontsize=16, fontweight='bold', y=0.995)

    commodities_order = ['Gold', 'WTI', 'Wheat', 'NaturalGas', 'Copper', 'Lithium']

    for idx, name in enumerate(commodities_order):
        if name not in merged_data:
            continue

        df = merged_data[name]

        # Select key features for correlation
        corr_features = ['Return', 'Vol_5', 'Return_lag1', 'GPRD',
                         'geo_keyword_hits', 'sentiment']

        # Filter only existing columns
        existing_features = [f for f in corr_features if f in df.columns]
        corr_df = df[existing_features].corr()

        # Plot
        ax = axes[idx // 3, idx % 3]
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(APPENDIX_DIR, 'appendix_f_correlation_heatmaps.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path}\n")
    plt.close()


# =============================================================================
# APPENDIX FIGURE 2: VOLATILITY WITH EVENTS + GPRD
# =============================================================================

def plot_volatility_with_events(merged_data):
    """
    Plot Vol_5 with geopolitical event markers and GPRD overlay.
    Shows 3 key periods: 2008 crisis, COVID 2020, Ukraine 2022.
    """
    print("📊 Generating Appendix G: Volatility with Events...")

    events = {
        '2008 Crisis': ('2008-01-01', '2009-12-31'),
        'COVID-19': ('2020-01-01', '2020-12-31'),
        'Ukraine': ('2022-01-01', '2023-06-30')
    }

    commodities_to_plot = ['WTI', 'Wheat', 'Gold']

    for name in commodities_to_plot:
        if name not in merged_data:
            print(f"  ⚠️ Skipping {name} - not found")
            continue

        df = merged_data[name].copy()

        # Check required columns
        if 'Vol_5' not in df.columns or 'GPRD' not in df.columns:
            print(f"  ⚠️ Skipping {name} - missing Vol_5 or GPRD")
            continue

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Plot Vol_5 on primary axis
        color1 = 'tab:orange'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('5-Day Rolling Volatility (Vol_5)', color=color1, fontsize=12)
        ax1.plot(df['Date'], df['Vol_5'], color=color1, alpha=0.7,
                 linewidth=1.2, label='Vol_5')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.3)

        # Mark event periods
        for event_name, (start, end) in events.items():
            event_dates = df[(df['Date'] >= start) & (df['Date'] <= end)]['Date']
            if len(event_dates) > 0:
                ax1.axvspan(event_dates.min(), event_dates.max(),
                            alpha=0.15, color='red')

        # Plot GPRD on secondary axis
        ax2 = ax1.twinx()
        color2 = 'tab:purple'
        ax2.set_ylabel('Geopolitical Risk Index (GPRD)', color=color2, fontsize=12)
        ax2.plot(df['Date'], df['GPRD'], color=color2, alpha=0.5,
                 linewidth=1, label='GPRD')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title(f'{name}: Volatility Dynamics During Geopolitical Events',
                  fontsize=14, fontweight='bold')

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                   framealpha=0.9)

        plt.tight_layout()
        save_path = os.path.join(APPENDIX_DIR, f'appendix_g_volatility_{name.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
        plt.close()

    print()


# =============================================================================
# APPENDIX FIGURE 3: FEATURE DISTRIBUTIONS
# =============================================================================

def plot_feature_distributions(merged_data):
    """
    Create clean histogram grid showing feature distributions for all commodities.
    Emphasizes heavy-tailed returns and GPRD skewness.
    """
    print("📊 Generating Appendix H: Feature Distributions...")

    fig, axes = plt.subplots(6, 5, figsize=(20, 18))
    fig.suptitle('Feature Distributions Across Commodities',
                 fontsize=18, fontweight='bold', y=0.995)

    commodities_order = ['Gold', 'WTI', 'Wheat', 'NaturalGas', 'Copper', 'Lithium']
    features = ['Return', 'Vol_5', 'GPRD', 'geo_keyword_hits', 'sentiment']

    for i, name in enumerate(commodities_order):
        if name not in merged_data:
            continue

        df = merged_data[name]

        for j, feature in enumerate(features):
            ax = axes[i, j]

            if feature in df.columns:
                data = df[feature].dropna()

                # Plot histogram
                ax.hist(data, bins=50, color='steelblue', alpha=0.7, edgecolor='black')

                # Add mean line
                mean_val = data.mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                           label=f'μ={mean_val:.3f}')

                # Labels
                if i == 0:
                    ax.set_title(feature, fontsize=11, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(name, fontsize=10, fontweight='bold', rotation=0,
                                  ha='right', va='center')

                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.legend(fontsize=7, loc='upper right')
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_path = os.path.join(APPENDIX_DIR, 'appendix_h_feature_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {save_path}\n")
    plt.close()


# =============================================================================
# APPENDIX FIGURE 4: TIME SERIES (SIMPLIFIED 2-AXIS)
# =============================================================================

def plot_timeseries_simplified(merged_data, price_cols):
    """
    Simplified time series: Price + Vol_5 (axis 1), GPRD + geo_keyword_hits (axis 2).
    Shows 3 representative commodities to illustrate heterogeneity.
    """
    print("📊 Generating Appendix I: Time Series Evolution...")

    commodities_to_plot = ['WTI', 'Wheat', 'Copper']

    for name in commodities_to_plot:
        if name not in merged_data:
            print(f"  ⚠️ Skipping {name} - not found")
            continue

        df = merged_data[name].copy()
        price_col = price_cols.get(name)

        # Check required columns
        required = ['Date', 'Vol_5', 'GPRD', 'geo_keyword_hits']
        missing = [col for col in required if col not in df.columns]
        if missing or price_col not in df.columns:
            print(f"  ⚠️ Skipping {name} - missing columns: {missing + [price_col]}")
            continue

        fig, ax1 = plt.subplots(figsize=(16, 7))

        # --- Axis 1: Price and Vol_5 ---
        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price (normalized) | Volatility', color=color1,
                       fontsize=12, fontweight='bold')

        # Normalize price to 0-1 for better visualization
        price_min = df[price_col].min()
        price_max = df[price_col].max()
        if price_max > price_min:  # Avoid division by zero
            price_normalized = (df[price_col] - price_min) / (price_max - price_min)
        else:
            price_normalized = df[price_col]

        line1 = ax1.plot(df['Date'], price_normalized, color='blue', alpha=0.6,
                         linewidth=1.5, label='Price (normalized)')
        line2 = ax1.plot(df['Date'], df['Vol_5'], color='red', alpha=0.7,
                         linewidth=1.2, label='Vol_5')

        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.3)

        # --- Axis 2: GPRD and geo_keyword_hits ---
        ax2 = ax1.twinx()
        color2 = 'tab:purple'
        ax2.set_ylabel('Geopolitical Indicators', color=color2,
                       fontsize=12, fontweight='bold')

        line3 = ax2.plot(df['Date'], df['GPRD'], color='purple', alpha=0.5,
                         linewidth=1.2, label='GPRD')
        line4 = ax2.plot(df['Date'], df['geo_keyword_hits'], color='green',
                         alpha=0.5, linewidth=1, label='Geo News Hits')

        ax2.tick_params(axis='y', labelcolor=color2)

        # --- Title and Combined Legend ---
        plt.title(f'{name}: Price, Volatility & Geopolitical Indicators (2003-2024)',
                  fontsize=14, fontweight='bold', pad=20)

        # Combine all lines for legend
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.95)

        plt.tight_layout()
        save_path = os.path.join(APPENDIX_DIR, f'appendix_i_timeseries_{name.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: {save_path}")
        plt.close()

def plot_event_timeline_alignment(
    merged_data,
    price_cols,
    output_dir,
    commodity="WTI",
    events=None,
    spike_proxy="abs_return",   # "abs_return" or "vol_5"
    gprd_smooth_window=21,      # rolling mean to make GPRD peaks readable
    event_padding_days=60       # widen each event window for context
):
    """
    VISUAL: Timeline with major crises, commodity price spikes, and GPRD overlay.
    Also annotates the time delay between (price spike) and (GPRD peak) per event,
    highlighting why GPRD aligns but may lag (weak forecasting power).
    """
    if events is None:
        events = {
            "2008 Financial Crisis": ("2008-01-01", "2009-12-31"),
            "2020 COVID-19": ("2020-01-01", "2020-12-31"),
            "2022 Ukraine": ("2022-01-01", "2023-06-30"),
        }

    if commodity not in merged_data:
        print(f" ⚠️ Skipping timeline - {commodity} not found")
        return

    df = merged_data[commodity].copy()
    if "Date" not in df.columns:
        print(f" ⚠️ Skipping {commodity} - missing Date column")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    price_col = price_cols.get(commodity)
    required_cols = ["GPRD"]
    missing = [c for c in required_cols if c not in df.columns]
    if price_col is None or price_col not in df.columns or missing:
        print(f" ⚠️ Skipping {commodity} - missing columns: {missing + [price_col]}")
        return

    # Build spike proxy
    if spike_proxy == "vol_5":
        if "Vol_5" not in df.columns:
            print(f" ⚠️ spike_proxy='vol_5' requested but Vol_5 missing for {commodity}")
            return
        spike_series = df["Vol_5"].astype(float)
        spike_label = "Volatility spike (Vol_5)"
    else:
        # abs daily returns from price
        price = df[price_col].astype(float)
        ret = price.pct_change()
        spike_series = ret.abs()
        spike_label = "Price spike proxy (|daily return|)"

    # Normalize price for readability (0-1)
    price_raw = df[price_col].astype(float)
    pmin, pmax = price_raw.min(), price_raw.max()
    price_norm = (price_raw - pmin) / (pmax - pmin) if pmax > pmin else price_raw * 0.0

    # Smooth & normalize GPRD (0-1)
    gprd_raw = df["GPRD"].astype(float)
    gprd_sm = gprd_raw.rolling(gprd_smooth_window, min_periods=1).mean()
    gmin, gmax = gprd_sm.min(), gprd_sm.max()
    gprd_norm = (gprd_sm - gmin) / (gmax - gmin) if gmax > gmin else gprd_sm * 0.0

    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot normalized price
    ax.plot(df["Date"], price_norm, color="tab:blue", linewidth=1.8, alpha=0.85,
            label="Commodity price (normalized)")

    # Plot spike proxy (scaled) on same axis for visibility
    spike_scaled = (spike_series - spike_series.min()) / (spike_series.max() - spike_series.min()) \
        if spike_series.max() > spike_series.min() else spike_series * 0.0
    ax.plot(df["Date"], spike_scaled, color="tab:red", linewidth=1.2, alpha=0.6,
            label=spike_label)

    # Secondary axis for GPRD (normalized, but on right axis for interpretability)
    ax2 = ax.twinx()
    ax2.plot(df["Date"], gprd_norm, color="tab:purple", linewidth=1.5, alpha=0.65,
             label=f"GPRD (smoothed {gprd_smooth_window}d, normalized)")

    # Shade events + compute lag between spike peak and GPRD peak within each window
    annotations = []
    for event_name, (start, end) in events.items():
        start_dt = pd.to_datetime(start) - pd.Timedelta(days=event_padding_days)
        end_dt = pd.to_datetime(end) + pd.Timedelta(days=event_padding_days)

        mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt)
        if mask.sum() < 10:
            continue

        ax.axvspan(start_dt, end_dt, color="grey", alpha=0.12)
        ax.text(start_dt, 1.02, event_name, fontsize=9, fontweight="bold",
                ha="left", va="bottom", transform=ax.get_xaxis_transform())

        window = df.loc[mask, ["Date"]].copy()
        window["spike"] = spike_series.loc[mask].values
        window["gprd"] = gprd_sm.loc[mask].values

        spike_peak_idx = window["spike"].idxmax()
        gprd_peak_idx = window["gprd"].idxmax()

        spike_peak_date = window.loc[spike_peak_idx, "Date"]
        gprd_peak_date = window.loc[gprd_peak_idx, "Date"]
        lag_days = (gprd_peak_date - spike_peak_date).days

        annotations.append((event_name, spike_peak_date, gprd_peak_date, lag_days))

        # Mark peaks
        ax.scatter([spike_peak_date], [spike_scaled.loc[spike_peak_idx]],
                   color="tab:red", s=45, zorder=5)
        ax2.scatter([gprd_peak_date], [gprd_norm.loc[gprd_peak_idx]],
                    color="tab:purple", s=45, zorder=5)

        # Draw an arrow showing delay
        y_arrow = 0.95  # in axis fraction
        ax.annotate(
            "",
            xy=(gprd_peak_date, y_arrow),
            xytext=(spike_peak_date, y_arrow),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.2, alpha=0.8),
        )
        ax.text(
            spike_peak_date + (gprd_peak_date - spike_peak_date) / 2,
            y_arrow + 0.03,
            f"Δ {lag_days}d",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # Axis formatting
    ax.set_title(
        f"{commodity}: Crisis Timeline vs Price Spikes and GPRD (Alignment + Lag)",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized series (price + spike proxy)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("GPRD (smoothed + normalized)", fontsize=11, fontweight="bold")

    ax.grid(alpha=0.25)
    ax.set_ylim(-0.05, 1.10)
    ax2.set_ylim(-0.05, 1.10)

    # Combined legend
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="lower right", framealpha=0.95, fontsize=9)

    # Footer note
    ax.text(
        0.01, -0.12,
        "Interpretation: GPRD often rises around crises (alignment) but peaks after market moves (lag), limiting forecasting power.",
        transform=ax.transAxes, fontsize=9
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"appendix_j_timeline_lag_{commodity.lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" ✅ Saved: {save_path}")

    # Optional console summary
    if annotations:
        print("Lag summary (GPRD peak date - spike peak date):")
        for e, sdt, gdt, lag in annotations:
            print(f"  {e:22s}: spike={sdt.date()} | gprd={gdt.date()} | Δ={lag:+d} days")

    plt.close()



# =============================================================================
# ORIGINAL EDA FUNCTIONS (KEPT FOR REFERENCE)
# =============================================================================

def run_original_eda(merged_data, price_cols):
    """Original EDA code from your script - kept for reference."""
    print("📊 Running Original EDA...")

    for name, df in merged_data.items():
        print(f"\n--- EDA for {name} ---")

        # Basic statistics
        eda_cols = ['Return', 'MA_5', 'Vol_5', 'Return_lag1', 'GPRD',
                    'geo_keyword_hits', 'sentiment']
        existing_cols = [c for c in eda_cols if c in df.columns]
        if existing_cols:
            print(df[existing_cols].describe())
            print("\nCorrelation matrix:\n", df[existing_cols].corr())

        # Stationarity tests
        if price_cols[name] in df.columns:
            print("\nADF Test for Price:")
            adf_result = adfuller(df[price_cols[name]].dropna())
            print(f"  ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3e}")
            print(f"  {'Stationary' if adf_result[1] < 0.05 else 'Not stationary'}")

        if 'Return' in df.columns:
            print("ADF Test for Return:")
            adf_result = adfuller(df['Return'].dropna())
            print(f"  ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3e}")
            print(f"  {'Stationary' if adf_result[1] < 0.05 else 'Not stationary'}")

        if 'Vol_5' in df.columns:
            print("ADF Test for Vol_5:")
            adf_result = adfuller(df['Vol_5'].dropna())
            print(f"  ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3e}")
            print(f"  {'Stationary' if adf_result[1] < 0.05 else 'Not stationary'}")

    print("\n✅ Original EDA complete\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("PUBLICATION-QUALITY APPENDIX FIGURE GENERATOR")
    print("=" * 70)
    print()

    # Load data
    merged_data = load_enriched_data()

    if not merged_data:
        print("❌ No data loaded. Please check your enriched data files.")
        return

    # Generate 4 publication-quality appendix figures
    print("🎨 Generating publication-quality appendix figures...\n")

    try:
        plot_correlation_heatmaps(merged_data)
    except Exception as e:
        print(f"  ❌ Error in correlation heatmaps: {e}\n")

    try:
        plot_volatility_with_events(merged_data)
    except Exception as e:
        print(f"  ❌ Error in volatility plots: {e}\n")

    try:
        plot_feature_distributions(merged_data)
    except Exception as e:
        print(f"  ❌ Error in distributions: {e}\n")

    try:
        plot_timeseries_simplified(merged_data, price_cols)
    except Exception as e:
        print(f"  ❌ Error in time series: {e}\n")

    try:
        # Appendix J: Timeline alignment + lag (presentation-style)
        plot_event_timeline_alignment(merged_data, price_cols, APPENDIX_DIR, commodity="WTI")
        plot_event_timeline_alignment(merged_data, price_cols, APPENDIX_DIR, commodity="Wheat")
        plot_event_timeline_alignment(merged_data, price_cols, APPENDIX_DIR, commodity="Gold")
    except Exception as e:
        print(f" ❌ Error in timeline alignment plot: {e}\n")

    print("=" * 70)
    print(f"✅ ALL APPENDIX FIGURES SAVED TO: {APPENDIX_DIR}")
    print("=" * 70)
    print("\nReference in text:")
    print("  - Appendix F: Correlation matrices (multicollinearity evidence)")
    print("  - Appendix G: Volatility during geopolitical events")
    print("  - Appendix H: Feature distributions (heavy tails, skewness)")
    print("  - Appendix I: Time-series evolution by commodity class")
    print()


if __name__ == "__main__":
    main()
