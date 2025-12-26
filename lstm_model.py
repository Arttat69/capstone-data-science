

# Enhanced LSTM Volatility Forecasting with Feature Comparison
# Tests: Baseline, GPRD, geo_keyword_hits, Combined, and Granger-optimized lags

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Configuration
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
MODEL_RESULTS_DIR = os.path.join(DATA_DIR, 'model_results')
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)

commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]

# Granger causality optimal lags from analysis
GRANGER_LAGS = {
    'Gold': {'geo': 3, 'gprd': None},
    'WTI': {'geo': None, 'gprd': None},
    'Wheat': {'geo': None, 'gprd': None},
    'NaturalGas': {'geo': 2, 'gprd': None},
    'Copper': {'geo': None, 'gprd': 1},
    'Lithium': {'geo': 1, 'gprd': None}
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sequences(data, feature_cols, target_col, seq_length):
    """Create sequences for LSTM"""
    xs, ys = [], []
    data_reset = data.reset_index(drop=True)

    for i in range(len(data_reset) - seq_length):
        x = data_reset[feature_cols].iloc[i:i + seq_length].values
        y = data_reset[target_col].iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def compute_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)


def backtest_volatility_strategy(vol_pred, vol_actual, returns, transaction_cost=0.001):
    """Trading strategy based on volatility predictions"""
    high_vol_threshold = np.percentile(vol_pred, 75)
    low_vol_threshold = np.percentile(vol_pred, 25)

    signals = np.zeros(len(vol_pred))
    signals[vol_pred > high_vol_threshold] = -0.5
    signals[vol_pred < low_vol_threshold] = 1.0
    signals[(vol_pred >= low_vol_threshold) & (vol_pred <= high_vol_threshold)] = 0.5

    aligned_returns = returns[1:len(signals) + 1]
    signals = signals[:len(aligned_returns)]

    strategy_returns = signals * aligned_returns
    position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
    transaction_costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - transaction_costs[:len(strategy_returns)]

    sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-10) * np.sqrt(252)
    cumulative_return = np.prod(1 + strategy_returns) - 1
    max_dd = compute_max_drawdown(strategy_returns)
    win_rate = np.mean(strategy_returns > 0)

    bh_returns = aligned_returns
    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-10) * np.sqrt(252)

    return {
        'sharpe': sharpe,
        'cumulative_return': cumulative_return * 100,
        'max_drawdown': max_dd * 100,
        'win_rate': win_rate * 100,
        'bh_sharpe': bh_sharpe,
        'alpha': sharpe - bh_sharpe,
        'avg_return_pct': np.mean(strategy_returns) * 100,
        'volatility': np.std(strategy_returns) * 100
    }


def build_lstm_model(input_shape):
    """Build LSTM with regularization"""
    n_features = input_shape[1]

    if n_features > 15:
        # Smaller network for high-dimensional input
        model = Sequential([
            Input(shape=input_shape),
            LSTM(24, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
    else:
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


def plot_results(commodity, y_true, y_pred, returns, history, feature_set_name):
    """Visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{commodity} - LSTM Volatility Prediction ({feature_set_name})', fontsize=16, fontweight='bold')

    # Plot 1: Predictions vs Actual
    plot_range = min(200, len(y_true))
    axes[0, 0].plot(y_true[-plot_range:], label='Actual Vol', alpha=0.7, linewidth=2, color='blue')
    axes[0, 0].plot(y_pred[-plot_range:], label='Predicted Vol', alpha=0.7, linewidth=2, color='red')
    axes[0, 0].set_title(f'Volatility Forecast (Last {plot_range} days)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Volatility')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Training history
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Training History')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (Huber)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Residuals
    residuals = y_true - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].set_xlabel('Predicted Volatility')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Distribution comparison
    try:
        n_bins = min(30, max(10, int(len(y_true) / 50)))
        axes[1, 1].hist(y_true, bins=n_bins, alpha=0.6, label='Actual', density=True, color='blue')
        axes[1, 1].hist(y_pred, bins=n_bins, alpha=0.6, label='Predicted', density=True, color='red')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].set_xlabel('Volatility')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f'Distribution plot unavailable',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Distribution Comparison')

    plt.tight_layout()
    safe_name = feature_set_name.replace(' ', '_').replace('+', 'and')
    plt.savefig(os.path.join(MODEL_RESULTS_DIR, f'{commodity}_lstm_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN MODELING FUNCTION
# ============================================================================

def run_lstm_volatility(commodity_name, df, feature_set='baseline'):
    """
    Run LSTM volatility prediction with different feature sets

    feature_set options:
    - 'baseline': Only Return and Vol_5 lags
    - 'gprd': + GPRD features
    - 'geo': + geo_keyword_hits features
    - 'combined': GPRD + geo_keyword_hits + sentiment
    - 'granger': Uses Granger-optimal lags for geo_keyword_hits
    """
    print(f"\n{'=' * 80}")
    print(f"Processing: {commodity_name} | Feature Set: {feature_set.upper()}")
    print(f"{'=' * 80}")

    df = df.copy().sort_values('Date').reset_index(drop=True)

    # Find Close column
    close_col = None
    for col in df.columns:
        if 'Close' in col or 'close' in col:
            close_col = col
            break

    if close_col is None:
        print(f"ERROR: No Close column found")
        return None

    if close_col != 'Close':
        df['Close'] = df[close_col]

    # Calculate basic features
    if 'Return' not in df.columns:
        df['Return'] = df['Close'].pct_change()

    if 'Vol_5' not in df.columns:
        df['Vol_5'] = df['Return'].rolling(5).std()

    # Base features
    base_features = ['Return', 'Vol_5']
    features_to_lag = base_features.copy()
    max_lag = 3  # Default lag depth

    # Add features based on feature_set
    if feature_set == 'baseline':
        print(f"  Using BASELINE features only (Return, Vol_5)")

    elif feature_set == 'gprd':
        if 'GPRD' in df.columns:
            df['GPRD'].fillna(method='ffill', inplace=True)
            df['GPRD'].fillna(0, inplace=True)
            df['GPRD_ma5'] = df['GPRD'].rolling(5).mean()
            features_to_lag.extend(['GPRD', 'GPRD_ma5'])
            print(f"  Added GPRD features")
        else:
            print(f"  WARNING: GPRD not found, using baseline")

    elif feature_set == 'geo':
        if 'geo_keyword_hits' in df.columns:
            df['geo_keyword_hits'].fillna(0, inplace=True)
            df['geo_ma5'] = df['geo_keyword_hits'].rolling(5).mean()
            features_to_lag.extend(['geo_keyword_hits', 'geo_ma5'])
            print(f"  Added geo_keyword_hits features")
        else:
            print(f"  WARNING: geo_keyword_hits not found, using baseline")

    elif feature_set == 'combined':
        added = []
        if 'GPRD' in df.columns:
            df['GPRD'].fillna(method='ffill', inplace=True)
            df['GPRD'].fillna(0, inplace=True)
            features_to_lag.append('GPRD')
            added.append('GPRD')

        if 'geo_keyword_hits' in df.columns:
            df['geo_keyword_hits'].fillna(0, inplace=True)
            features_to_lag.append('geo_keyword_hits')
            added.append('geo_keyword_hits')

        if 'sentiment' in df.columns:
            df['sentiment'].fillna(0, inplace=True)
            features_to_lag.append('sentiment')
            added.append('sentiment')

        print(f"  Added COMBINED features: {', '.join(added)}")

    elif feature_set == 'granger':
        # Use Granger-optimal lags
        granger_info = GRANGER_LAGS.get(commodity_name, {})

        if granger_info.get('geo') and 'geo_keyword_hits' in df.columns:
            optimal_lag = granger_info['geo']
            df['geo_keyword_hits'].fillna(0, inplace=True)
            # Only use the optimal lag
            df[f'geo_lag{optimal_lag}'] = df['geo_keyword_hits'].shift(optimal_lag)
            features_to_lag.append(f'geo_lag{optimal_lag}')
            print(f"  Using GRANGER-OPTIMAL: geo_keyword_hits lag {optimal_lag}")
        elif granger_info.get('gprd') and 'GPRD' in df.columns:
            optimal_lag = granger_info['gprd']
            df['GPRD'].fillna(method='ffill', inplace=True)
            df['GPRD'].fillna(0, inplace=True)
            df[f'GPRD_lag{optimal_lag}'] = df['GPRD'].shift(optimal_lag)
            features_to_lag.append(f'GPRD_lag{optimal_lag}')
            print(f"  Using GRANGER-OPTIMAL: GPRD lag {optimal_lag}")
        else:
            print(f"  WARNING: No Granger-significant features for {commodity_name}, using baseline")

    # Create lagged features (skip if already created in granger mode)
    if feature_set != 'granger':
        feature_cols = []
        for feat in features_to_lag:
            for lag in range(1, max_lag + 1):
                df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
                feature_cols.append(f'{feat}_lag{lag}')
    else:
        # For granger mode, we manually added the optimal lag
        feature_cols = [col for col in df.columns if 'lag' in col and col in features_to_lag]
        # Also add base lags
        for feat in base_features:
            for lag in range(1, max_lag + 1):
                df[f'{feat}_lag{lag}'] = df[feat].shift(lag)
                feature_cols.append(f'{feat}_lag{lag}')

    print(f"  Total features: {len(feature_cols)}")

    # Clean data
    required_cols = ['Date', 'Vol_5', 'Return'] + feature_cols
    df_clean = df[required_cols].dropna().reset_index(drop=True)

    print(f"  Clean data: {len(df_clean)} rows")

    if len(df_clean) < 100:
        print("  ERROR: Insufficient data")
        return None

    # Train/test split
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    if len(train_df) < 50 or len(test_df) < 20:
        print("  ERROR: Insufficient train/test data")
        return None

    # Scale features
    scaler = RobustScaler()

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Rebuild as DataFrames
    train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

    train_scaled['Vol_5'] = train_df['Vol_5'].values
    test_scaled['Vol_5'] = test_df['Vol_5'].values
    train_scaled['Return'] = train_df['Return'].values
    test_scaled['Return'] = test_df['Return'].values

    # Create sequences
    seq_length = 15
    X_train_seq, y_train_seq = create_sequences(train_scaled, feature_cols, 'Vol_5', seq_length)
    X_test_seq, y_test_seq = create_sequences(test_scaled, feature_cols, 'Vol_5', seq_length)

    print(f"  Sequences - Train: {X_train_seq.shape} | Test: {X_test_seq.shape}")

    if len(X_train_seq) < 20 or len(X_test_seq) < 5:
        print("  ERROR: Insufficient sequences")
        return None

    # Build and train model
    model = build_lstm_model(input_shape=(seq_length, len(feature_cols)))

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=0)

    batch_size = 32 if len(feature_cols) > 15 else 64

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Predictions
    y_pred_train = model.predict(X_train_seq, verbose=0).squeeze()
    y_pred_test = model.predict(X_test_seq, verbose=0).squeeze()

    # Metrics
    train_r2 = r2_score(y_train_seq, y_pred_train)
    test_r2 = r2_score(y_test_seq, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_seq, y_pred_test))
    test_mae = mean_absolute_error(y_test_seq, y_pred_test)

    print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

    # Backtest
    test_returns = test_scaled['Return'].iloc[seq_length:].values
    backtest_results = backtest_volatility_strategy(y_pred_test, y_test_seq, test_returns)

    print(f"  Sharpe: {backtest_results['sharpe']:.3f} | Alpha: {backtest_results['alpha']:.3f}")
    print(
        f"  Cum Return: {backtest_results['cumulative_return']:.2f}% | Max DD: {backtest_results['max_drawdown']:.2f}%")

    results = {
        'commodity': commodity_name,
        'feature_set': feature_set,
        'n_features': len(feature_cols),
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        **backtest_results
    }

    # Plot results
    plot_results(commodity_name, y_test_seq, y_pred_test, test_returns, history, feature_set)

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("ENHANCED LSTM VOLATILITY PREDICTION")
    print("Testing: Baseline | GPRD | geo_keyword_hits | Combined | Granger-Optimal")
    print("=" * 80)

    print("\nLoading enriched data...")
    merged_data = {}

    for name in commodities:
        fname = f"{name.lower()}_enriched.csv"
        path = os.path.join(ENRICHED_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            merged_data[name] = df
            has_gprd = 'GPRD' in df.columns
            has_geo = 'geo_keyword_hits' in df.columns
            print(f"  ✓ {name}: {len(df)} rows | GPRD: {'✓' if has_gprd else '✗'} | geo: {'✓' if has_geo else '✗'}")
        else:
            print(f"  ✗ Missing: {name}")

    all_results = []

    # Test all feature sets for each commodity
    feature_sets = ['baseline', 'gprd', 'geo', 'combined', 'granger']

    for commodity in merged_data.keys():
        for feature_set in feature_sets:
            try:
                results = run_lstm_volatility(commodity, merged_data[commodity], feature_set=feature_set)
                if results:
                    all_results.append(results)
            except Exception as e:
                print(f"  ERROR in {commodity} - {feature_set}: {str(e)}")
                continue

    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(MODEL_RESULTS_DIR, 'lstm_volatility_feature_comparison.csv'), index=False)

        print("\n" + "=" * 80)
        print("SUMMARY: Feature Set Performance Comparison")
        print("=" * 80)

        # Show key metrics
        display_cols = ['commodity', 'feature_set', 'test_r2', 'test_rmse', 'sharpe', 'alpha', 'cumulative_return']
        print(results_df[display_cols].to_string(index=False))

        # Compare average performance by feature set
        print("\n" + "=" * 80)
        print("Average Performance by Feature Set")
        print("=" * 80)
        comparison = results_df.groupby('feature_set')[['test_r2', 'test_rmse', 'sharpe', 'alpha']].mean()
        print(comparison)

        # Show best feature set per commodity
        print("\n" + "=" * 80)
        print("Best Feature Set by Commodity (by Sharpe Ratio)")
        print("=" * 80)
        best_by_commodity = results_df.loc[results_df.groupby('commodity')['sharpe'].idxmax()]
        print(best_by_commodity[['commodity', 'feature_set', 'sharpe', 'test_r2', 'alpha']].to_string(index=False))

        # Statistical comparison: baseline vs others
        print("\n" + "=" * 80)
        print("Improvement Over Baseline")
        print("=" * 80)

        for commodity in results_df['commodity'].unique():
            subset = results_df[results_df['commodity'] == commodity]
            baseline = subset[subset['feature_set'] == 'baseline']

            if len(baseline) > 0:
                baseline_sharpe = baseline.iloc[0]['sharpe']
                baseline_r2 = baseline.iloc[0]['test_r2']

                print(f"\n{commodity}:")
                print(f"  Baseline - Sharpe: {baseline_sharpe:.3f}, R²: {baseline_r2:.4f}")

                for feature_set in ['gprd', 'geo', 'combined', 'granger']:
                    fs_data = subset[subset['feature_set'] == feature_set]
                    if len(fs_data) > 0:
                        sharpe_imp = fs_data.iloc[0]['sharpe'] - baseline_sharpe
                        r2_imp = fs_data.iloc[0]['test_r2'] - baseline_r2
                        symbol = '✓' if sharpe_imp > 0 else '✗'
                        print(f"  {symbol} {feature_set:10s}: Sharpe {sharpe_imp:+.3f}, R² {r2_imp:+.4f}")

        # Create comparison visualizations
        print("\n" + "=" * 80)
        print("Generating comparison visualizations...")
        print("=" * 80)

        # Plot 1: Sharpe Ratio Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Sharpe by feature set
        ax = axes[0, 0]
        pivot_sharpe = results_df.pivot(index='commodity', columns='feature_set', values='sharpe')
        pivot_sharpe.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Sharpe Ratio by Feature Set', fontsize=13, fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('Commodity')
        ax.legend(title='Feature Set', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # R² by feature set
        ax = axes[0, 1]
        pivot_r2 = results_df.pivot(index='commodity', columns='feature_set', values='test_r2')
        pivot_r2.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Test R² by Feature Set', fontsize=13, fontweight='bold')
        ax.set_ylabel('R² Score')
        ax.set_xlabel('Commodity')
        ax.legend(title='Feature Set', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # RMSE by feature set
        ax = axes[1, 0]
        pivot_rmse = results_df.pivot(index='commodity', columns='feature_set', values='test_rmse')
        pivot_rmse.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Test RMSE by Feature Set (Lower is Better)', fontsize=13, fontweight='bold')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Commodity')
        ax.legend(title='Feature Set', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Alpha (excess return over buy-hold)
        ax = axes[1, 1]
        pivot_alpha = results_df.pivot(index='commodity', columns='feature_set', values='alpha')
        pivot_alpha.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Alpha vs Buy-and-Hold by Feature Set', fontsize=13, fontweight='bold')
        ax.set_ylabel('Alpha')
        ax.set_xlabel('Commodity')
        ax.legend(title='Feature Set', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_RESULTS_DIR, 'feature_set_comparison_summary.png'),
                    dpi=300, bbox_inches='tight')
        print("  ✓ Saved: feature_set_comparison_summary.png")
        plt.close()

        # Plot 2: Heatmap of improvements over baseline
        print("  Creating improvement heatmap...")

        improvement_data = []
        for commodity in results_df['commodity'].unique():
            subset = results_df[results_df['commodity'] == commodity]
            baseline = subset[subset['feature_set'] == 'baseline']

            if len(baseline) > 0:
                baseline_sharpe = baseline.iloc[0]['sharpe']
                row = {'Commodity': commodity}

                for feature_set in ['gprd', 'geo', 'combined', 'granger']:
                    fs_data = subset[subset['feature_set'] == feature_set]
                    if len(fs_data) > 0:
                        improvement = fs_data.iloc[0]['sharpe'] - baseline_sharpe
                        row[feature_set] = improvement
                    else:
                        row[feature_set] = np.nan

                improvement_data.append(row)

        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            improvement_df.set_index('Commodity', inplace=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(improvement_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                        ax=ax, cbar_kws={'label': 'Sharpe Improvement'}, linewidths=0.5)
            ax.set_title('Sharpe Ratio Improvement Over Baseline\n(Green = Better, Red = Worse)',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Set', fontsize=12)
            ax.set_ylabel('Commodity', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_RESULTS_DIR, 'sharpe_improvement_heatmap.png'),
                        dpi=300, bbox_inches='tight')
            print("  ✓ Saved: sharpe_improvement_heatmap.png")
            plt.close()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {MODEL_RESULTS_DIR}")
        print("  - lstm_volatility_feature_comparison.csv")
        print("  - feature_set_comparison_summary.png")
        print("  - sharpe_improvement_heatmap.png")
        print("  - Individual commodity plots for each feature set")

    else:
        print("\n✗ No successful results to save!")


if __name__ == "__main__":
    main()