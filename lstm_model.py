# Enhanced LSTM Commodity Forecasting - FIXED VERSION
# Handles missing enrichment data and provides proper backtesting

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_sequences(data, feature_cols, target_col, seq_length):
    """Create sequences for LSTM - fixed to use iloc consistently"""
    xs, ys = [], []
    data_reset = data.reset_index(drop=True)  # Reset index to avoid issues

    for i in range(len(data_reset) - seq_length):
        x = data_reset[feature_cols].iloc[i:i+seq_length].values
        y = data_reset[target_col].iloc[i+seq_length]
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

    # Align returns
    aligned_returns = returns[1:len(signals)+1]
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

def build_robust_lstm(input_shape):
    """Build LSTM with regularization"""
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

def plot_results(commodity, y_true, y_pred, returns, history, use_gprd):
    """Visualization with error handling"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{commodity} - LSTM Results (GPRD: {"ON" if use_gprd else "OFF"})', fontsize=16)

    # Plot 1: Predictions vs Actual
    plot_range = min(200, len(y_true))
    axes[0, 0].plot(y_true[-plot_range:], label='Actual Vol', alpha=0.7, linewidth=2)
    axes[0, 0].plot(y_pred[-plot_range:], label='Predicted Vol', alpha=0.7, linewidth=2)
    axes[0, 0].set_title(f'Volatility Forecast (Last {plot_range} days)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: Training history
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Training History')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Residuals
    residuals = y_true - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.3)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Distribution comparison (with error handling)
    try:
        # Auto-calculate bins based on data range
        n_bins = min(30, max(10, int(len(y_true) / 50)))
        axes[1, 1].hist(y_true, bins=n_bins, alpha=0.5, label='Actual', density=True)
        axes[1, 1].hist(y_pred, bins=n_bins, alpha=0.5, label='Predicted', density=True)
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    except Exception as e:
        # If histogram fails, plot KDE instead
        axes[1, 1].text(0.5, 0.5, f'Distribution plot unavailable\n({str(e)[:50]})',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Distribution Comparison')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_RESULTS_DIR, f'{commodity}_lstm_{"gprd" if use_gprd else "nogprd"}.png'), dpi=150)
    plt.close()

def main():
    # ============================================================================
    # MAIN MODELING PIPELINE
    # ============================================================================

    def run_enhanced_lstm_with_gprd(commodity_name, df, use_gprd=True):
        """Run LSTM with optional GPRD integration"""
        print(f"\n{'='*80}")
        print(f"Processing: {commodity_name} (GPRD: {'ON' if use_gprd else 'OFF'})")
        print(f"{'='*80}")

        df = df.copy().sort_values('Date').reset_index(drop=True)

        # Find and rename Close column
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

        # ============= NEW: GPRD FEATURE ENGINEERING =============
        # Base features to lag
        base_features = ['Return', 'Vol_5']

        # Add GPRD features if available and user requests them
        gprd_features = []
        if use_gprd:
            if 'GPRD' in df.columns:
                df['GPRD'].fillna(method='ffill', inplace=True)
                df['GPRD'].fillna(0, inplace=True)

                # SIMPLIFIED: Only use 2 most important GPRD features to save memory
                df['GPRD_ma5'] = df['GPRD'].rolling(5).mean()
                df['GPRD_high_regime'] = (df['GPRD'] > df['GPRD'].rolling(60).quantile(0.75)).astype(int)

                gprd_features = ['GPRD', 'GPRD_ma5', 'GPRD_high_regime']
                print(f"  Added GPRD features: {gprd_features}")

            if 'geo_keyword_hits' in df.columns:
                df['geo_keyword_hits'].fillna(0, inplace=True)
                gprd_features.append('geo_keyword_hits')

            if 'sentiment' in df.columns:
                df['sentiment'].fillna(0, inplace=True)
                gprd_features.append('sentiment')

        # Combine all features to lag
        features_to_lag = base_features + gprd_features

        # REDUCED LAG DEPTH: Only lag 1-3 instead of 1-5 to save memory
        max_lag = 3
        for feat in features_to_lag:
            for lag in range(1, max_lag + 1):
                df[f'{feat}_lag{lag}'] = df[feat].shift(lag)

        # Build feature column list
        feature_cols = []
        for feat in features_to_lag:
            for lag in range(1, max_lag + 1):
                feature_cols.append(f'{feat}_lag{lag}')

        print(f"  Total features: {len(feature_cols)} ({'with GPRD' if use_gprd else 'baseline'})")

        # Clean data
        required_cols = ['Date', 'Vol_5', 'Return'] + feature_cols
        df_clean = df[required_cols].dropna().reset_index(drop=True)

        print(f"Clean data: {len(df_clean)} rows, {len(feature_cols)} features")

        if len(df_clean) < 100:
            print("Insufficient data")
            return None

        # Train/test split
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx].copy()
        test_df = df_clean.iloc[split_idx:].copy()

        print(f"Train: {len(train_df)} | Test: {len(test_df)}")

        if len(train_df) < 50 or len(test_df) < 20:
            print("Insufficient train/test data")
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
        seq_length = 15  # Reduced from 20 to save memory
        X_train_seq, y_train_seq = create_sequences(train_scaled, feature_cols, 'Vol_5', seq_length)
        X_test_seq, y_test_seq = create_sequences(test_scaled, feature_cols, 'Vol_5', seq_length)

        print(f"Sequences - Train: {X_train_seq.shape} | Test: {X_test_seq.shape}")

        if len(X_train_seq) < 20 or len(X_test_seq) < 5:
            print("Insufficient sequences")
            return None

        # Build model with adjusted size for more features
        if len(feature_cols) > 15:
            # Smaller network for high-dimensional input
            model = Sequential([
                Input(shape=(seq_length, len(feature_cols))),
                LSTM(24, kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
                Dropout(0.2),
                Dense(8, activation='relu'),
                Dense(1)
            ])
        else:
            model = build_robust_lstm(input_shape=(seq_length, len(feature_cols)))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])

        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=0)

        # Reduced batch size for memory efficiency
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

        print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

        # Backtest
        test_returns = test_scaled['Return'].iloc[seq_length:].values
        backtest_results = backtest_volatility_strategy(y_pred_test, y_test_seq, test_returns)

        print(f"Sharpe: {backtest_results['sharpe']:.3f} | Alpha: {backtest_results['alpha']:.3f}")
        print(f"Cum Return: {backtest_results['cumulative_return']:.2f}% | Max DD: {backtest_results['max_drawdown']:.2f}%")

        results = {
            'commodity': commodity_name,
            'use_gprd': use_gprd,
            'n_features': len(feature_cols),
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            **backtest_results
        }

        plot_results(commodity_name, y_test_seq, y_pred_test, test_returns, history, use_gprd)

        return results

    # ============================================================================
    # RUN EXPERIMENTS
    # ============================================================================

    if __name__ == "__main__":
        print("Loading enriched data...")
        merged_data = {}

        for name in commodities:
            fname = f"{name.lower()}_enriched.csv"
            path = os.path.join(ENRICHED_DIR, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                merged_data[name] = df
                # Show ALL columns to verify GPRD is present
                has_gprd = 'GPRD' in df.columns
                print(f"  Loaded {name}: {len(df)} rows | GPRD: {'✓' if has_gprd else '✗'} | Cols: {len(df.columns)}")
            else:
                print(f"  Missing: {name}")

        all_results = []

        for commodity in merged_data.keys():
            # Run WITHOUT GPRD (baseline)
            print(f"\n{'=' * 80}")
            print(f"BASELINE RUN: {commodity} (no GPRD)")
            print(f"{'=' * 80}")
            results_no_gprd = run_enhanced_lstm_with_gprd(commodity, merged_data[commodity], use_gprd=False)
            if results_no_gprd:
                all_results.append(results_no_gprd)

            # Run WITH GPRD (if available)
            if 'GPRD' in merged_data[commodity].columns:
                print(f"\n{'=' * 80}")
                print(f"ENHANCED RUN: {commodity} (with GPRD)")
                print(f"{'=' * 80}")
                results_with_gprd = run_enhanced_lstm_with_gprd(commodity, merged_data[commodity], use_gprd=True)
                if results_with_gprd:
                    all_results.append(results_with_gprd)
            else:
                print(f"  Skipping GPRD test for {commodity} - no GPRD column found")

        # Save results
        if len(all_results) > 0:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(os.path.join(MODEL_RESULTS_DIR, 'lstm_gprd_comparison.csv'), index=False)

            print("\n" + "=" * 80)
            print("SUMMARY: GPRD Impact Analysis")
            print("=" * 80)
            print(
                results_df[['commodity', 'use_gprd', 'n_features', 'test_r2', 'sharpe', 'alpha', 'cumulative_return']])

            # Compare performance by GPRD usage
            print("\n" + "=" * 80)
            print("Average Performance: WITH vs WITHOUT GPRD")
            print("=" * 80)
            comparison = results_df.groupby('use_gprd')[['test_r2', 'sharpe', 'alpha', 'cumulative_return']].mean()
            print(comparison)

            # Show improvement per commodity
            print("\n" + "=" * 80)
            print("GPRD Improvement by Commodity")
            print("=" * 80)
            for commodity in results_df['commodity'].unique():
                subset = results_df[results_df['commodity'] == commodity]
                if len(subset) == 2:  # Has both baseline and GPRD
                    baseline = subset[subset['use_gprd'] == False].iloc[0]
                    gprd = subset[subset['use_gprd'] == True].iloc[0]
                    sharpe_improvement = gprd['sharpe'] - baseline['sharpe']
                    print(f"{commodity:12s}: Sharpe {baseline['sharpe']:.3f} → {gprd['sharpe']:.3f} "
                          f"({'↑' if sharpe_improvement > 0 else '↓'} {abs(sharpe_improvement):.3f})")
        else:
            print("\nNo successful results to save!")

if __name__ == "__main__":
    # This allows you to run this file individually for testing
    main()