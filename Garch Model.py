# GARCH Volatility Modeling with Geopolitical Risk Features
# Compares baseline GARCH vs GARCH with external regressors

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from arch.univariate import GARCH, ConstantMean, Normal
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configuration
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
GARCH_RESULTS_DIR = os.path.join(DATA_DIR, 'garch_results')
os.makedirs(GARCH_RESULTS_DIR, exist_ok=True)

commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]

print("=" * 80)
print("GARCH VOLATILITY MODELING WITH GEOPOLITICAL FEATURES")
print("=" * 80)
print("Models:")
print("  1. Baseline GARCH(1,1)")
print("  2. GARCH(1,1) + geo_keyword_hits")
print("  3. GARCH(1,1) + GPRD")
print("  4. GARCH(1,1) + geo_keyword_hits + GPRD")
print("=" * 80)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_returns(df):
    """Prepare returns series and external regressors"""
    # Find Close column
    close_col = None
    for col in df.columns:
        if 'Close' in col or 'close' in col:
            close_col = col
            break

    if close_col is None:
        return None, None, None

    # Calculate returns
    if 'Return' not in df.columns:
        df['Return'] = df[close_col].pct_change()

    # Convert returns to percentage (GARCH works better with percentage returns)
    returns = df['Return'] * 100

    # Prepare external regressors
    X_geo = None
    X_gprd = None

    if 'geo_keyword_hits' in df.columns:
        # Standardize geo_keyword_hits
        geo = df['geo_keyword_hits'].fillna(0)
        X_geo = (geo - geo.mean()) / geo.std()

    if 'GPRD' in df.columns:
        # Standardize GPRD
        gprd = df['GPRD'].fillna(method='ffill').fillna(0)
        X_gprd = (gprd - gprd.mean()) / gprd.std()

    return returns, X_geo, X_gprd


def fit_garch_model(returns, external_regressors=None, model_name="Baseline"):
    """
    Fit GARCH(1,1) model with optional external regressors

    Args:
        returns: Return series (in percentage)
        external_regressors: DataFrame of external variables (standardized)
        model_name: Name for logging

    Returns:
        Fitted model result
    """
    # Remove NaN values
    if external_regressors is not None:
        # Align returns and regressors
        valid_idx = returns.notna() & external_regressors.notna().all(axis=1)
        returns_clean = returns[valid_idx]
        X_clean = external_regressors[valid_idx]
    else:
        returns_clean = returns.dropna()
        X_clean = None

    if len(returns_clean) < 100:
        print(f"    ERROR: Insufficient data ({len(returns_clean)} obs)")
        return None

    # Build model
    try:
        if X_clean is not None and len(X_clean.columns) > 0:
            # GARCH with external regressors in variance equation
            model = arch_model(returns_clean, vol='GARCH', p=1, q=1,
                               x=X_clean, rescale=False)
        else:
            # Baseline GARCH
            model = arch_model(returns_clean, vol='GARCH', p=1, q=1, rescale=False)

        # Fit model
        result = model.fit(disp='off', show_warning=False)

        return result

    except Exception as e:
        print(f"    ERROR fitting {model_name}: {str(e)}")
        return None


def forecast_volatility(model_result, horizon=5):
    """
    Generate volatility forecast

    Args:
        model_result: Fitted GARCH model
        horizon: Forecast horizon in days

    Returns:
        Forecast DataFrame
    """
    try:
        forecast = model_result.forecast(horizon=horizon, reindex=False)
        return forecast
    except:
        return None


def calculate_metrics(returns, fitted_vol, actual_vol=None):
    """
    Calculate performance metrics

    Args:
        returns: Actual returns
        fitted_vol: Fitted/forecasted volatility from model
        actual_vol: Actual realized volatility (if available)
    """
    # Align series
    common_idx = returns.index.intersection(fitted_vol.index)
    returns_aligned = returns.loc[common_idx]
    vol_aligned = fitted_vol.loc[common_idx]

    # Calculate squared returns as proxy for realized variance
    realized_var = returns_aligned ** 2

    # Mean Squared Error
    mse = np.mean((realized_var - vol_aligned ** 2) ** 2)

    # Log-likelihood (approximation)
    ll = -0.5 * np.sum(np.log(2 * np.pi * vol_aligned ** 2) + realized_var / vol_aligned ** 2)

    # QLIKE (Quasi-Likelihood)
    qlike = np.mean(realized_var / vol_aligned ** 2 - np.log(realized_var / vol_aligned ** 2) - 1)

    return {
        'MSE': mse,
        'LogLik': ll,
        'QLIKE': qlike
    }


# ============================================================================
# MAIN GARCH ANALYSIS
# ============================================================================

def run_garch_analysis(commodity_name, df):
    """
    Run complete GARCH analysis for a commodity
    """
    print(f"\n{'=' * 80}")
    print(f"COMMODITY: {commodity_name}")
    print(f"{'=' * 80}")

    # Prepare data
    returns, X_geo, X_gprd = prepare_returns(df)

    if returns is None:
        print(f"  ERROR: Could not prepare returns")
        return None

    # Remove NaN and extreme outliers
    returns = returns.dropna()
    returns = returns[(returns.abs() < returns.abs().quantile(0.99))]  # Remove extreme outliers

    print(f"  Data: {len(returns)} observations")
    print(f"  Return stats: Mean={returns.mean():.4f}%, Std={returns.std():.4f}%")

    # Split train/test (80/20)
    split_idx = int(len(returns) * 0.8)
    returns_train = returns.iloc[:split_idx]
    returns_test = returns.iloc[split_idx:]

    print(f"  Train: {len(returns_train)} | Test: {len(returns_test)}")

    # Prepare external regressors for train/test
    if X_geo is not None:
        X_geo_train = X_geo.iloc[:split_idx]
        X_geo_test = X_geo.iloc[split_idx:]
    else:
        X_geo_train = X_geo_test = None

    if X_gprd is not None:
        X_gprd_train = X_gprd.iloc[:split_idx]
        X_gprd_test = X_gprd.iloc[split_idx:]
    else:
        X_gprd_train = X_gprd_test = None

    results = []
    fitted_models = {}

    # Model 1: Baseline GARCH(1,1)
    print(f"\n  [1/4] Fitting Baseline GARCH(1,1)...")
    model_baseline = fit_garch_model(returns_train, external_regressors=None,
                                     model_name="Baseline")

    if model_baseline:
        # In-sample fit
        fitted_vol = model_baseline.conditional_volatility
        metrics_is = calculate_metrics(returns_train, fitted_vol)

        # Out-of-sample forecast
        try:
            # Rolling forecast on test set
            forecast_vols = []
            for i in range(len(returns_test)):
                # Refit model on expanding window
                train_window = returns.iloc[:split_idx + i]
                model_temp = fit_garch_model(train_window, external_regressors=None,
                                             model_name="Baseline_rolling")
                if model_temp:
                    forecast = model_temp.forecast(horizon=1, reindex=False)
                    forecast_vols.append(np.sqrt(forecast.variance.values[-1, 0]))
                else:
                    forecast_vols.append(np.nan)

            forecast_vol_series = pd.Series(forecast_vols, index=returns_test.index)
            metrics_oos = calculate_metrics(returns_test, forecast_vol_series)
        except:
            metrics_oos = {'MSE': np.nan, 'LogLik': np.nan, 'QLIKE': np.nan}

        print(f"    In-sample QLIKE: {metrics_is['QLIKE']:.4f}")
        print(f"    Out-of-sample QLIKE: {metrics_oos['QLIKE']:.4f}")
        print(f"    AIC: {model_baseline.aic:.2f} | BIC: {model_baseline.bic:.2f}")

        results.append({
            'commodity': commodity_name,
            'model': 'Baseline GARCH(1,1)',
            'AIC': model_baseline.aic,
            'BIC': model_baseline.bic,
            'LogLik': model_baseline.loglikelihood,
            'QLIKE_in_sample': metrics_is['QLIKE'],
            'QLIKE_out_of_sample': metrics_oos['QLIKE'],
            'n_params': len(model_baseline.params)
        })

        fitted_models['baseline'] = model_baseline

    # Model 2: GARCH + geo_keyword_hits
    if X_geo_train is not None:
        print(f"\n  [2/4] Fitting GARCH + geo_keyword_hits...")
        X_geo_df = pd.DataFrame({'geo': X_geo_train}, index=returns_train.index)
        model_geo = fit_garch_model(returns_train, external_regressors=X_geo_df,
                                    model_name="GEO")

        if model_geo:
            fitted_vol = model_geo.conditional_volatility
            metrics_is = calculate_metrics(returns_train, fitted_vol)

            # Out-of-sample forecast (simplified - using last model)
            try:
                forecast_vols = []
                for i in range(min(50, len(returns_test))):  # Limit to 50 for speed
                    train_window = returns.iloc[:split_idx + i]
                    X_window = pd.DataFrame({'geo': X_geo.iloc[:split_idx + i]},
                                            index=train_window.index)
                    model_temp = fit_garch_model(train_window, external_regressors=X_window,
                                                 model_name="GEO_rolling")
                    if model_temp:
                        forecast = model_temp.forecast(horizon=1, reindex=False)
                        forecast_vols.append(np.sqrt(forecast.variance.values[-1, 0]))
                    else:
                        forecast_vols.append(np.nan)

                forecast_vol_series = pd.Series(forecast_vols,
                                                index=returns_test.index[:len(forecast_vols)])
                metrics_oos = calculate_metrics(returns_test.iloc[:len(forecast_vols)],
                                                forecast_vol_series)
            except:
                metrics_oos = {'MSE': np.nan, 'LogLik': np.nan, 'QLIKE': np.nan}

            print(f"    In-sample QLIKE: {metrics_is['QLIKE']:.4f}")
            print(f"    Out-of-sample QLIKE: {metrics_oos['QLIKE']:.4f}")
            print(f"    AIC: {model_geo.aic:.2f} | BIC: {model_geo.bic:.2f}")

            # Check if geo coefficient is significant
            geo_param = None
            geo_pval = None
            for param_name in model_geo.params.index:
                if 'geo' in param_name.lower():
                    geo_param = model_geo.params[param_name]
                    geo_pval = model_geo.pvalues[param_name]
                    break

            if geo_param is not None:
                sig_marker = '✓' if geo_pval < 0.05 else '✗'
                print(f"    geo coefficient: {geo_param:.6f} (p={geo_pval:.3f}) {sig_marker}")

            results.append({
                'commodity': commodity_name,
                'model': 'GARCH + geo_keyword_hits',
                'AIC': model_geo.aic,
                'BIC': model_geo.bic,
                'LogLik': model_geo.loglikelihood,
                'QLIKE_in_sample': metrics_is['QLIKE'],
                'QLIKE_out_of_sample': metrics_oos['QLIKE'],
                'n_params': len(model_geo.params),
                'geo_coef': geo_param if geo_param is not None else np.nan,
                'geo_pval': geo_pval if geo_pval is not None else np.nan
            })

            fitted_models['geo'] = model_geo

    # Model 3: GARCH + GPRD
    if X_gprd_train is not None:
        print(f"\n  [3/4] Fitting GARCH + GPRD...")
        X_gprd_df = pd.DataFrame({'gprd': X_gprd_train}, index=returns_train.index)
        model_gprd = fit_garch_model(returns_train, external_regressors=X_gprd_df,
                                     model_name="GPRD")

        if model_gprd:
            fitted_vol = model_gprd.conditional_volatility
            metrics_is = calculate_metrics(returns_train, fitted_vol)

            # Out-of-sample forecast (simplified)
            try:
                forecast_vols = []
                for i in range(min(50, len(returns_test))):
                    train_window = returns.iloc[:split_idx + i]
                    X_window = pd.DataFrame({'gprd': X_gprd.iloc[:split_idx + i]},
                                            index=train_window.index)
                    model_temp = fit_garch_model(train_window, external_regressors=X_window,
                                                 model_name="GPRD_rolling")
                    if model_temp:
                        forecast = model_temp.forecast(horizon=1, reindex=False)
                        forecast_vols.append(np.sqrt(forecast.variance.values[-1, 0]))
                    else:
                        forecast_vols.append(np.nan)

                forecast_vol_series = pd.Series(forecast_vols,
                                                index=returns_test.index[:len(forecast_vols)])
                metrics_oos = calculate_metrics(returns_test.iloc[:len(forecast_vols)],
                                                forecast_vol_series)
            except:
                metrics_oos = {'MSE': np.nan, 'LogLik': np.nan, 'QLIKE': np.nan}

            print(f"    In-sample QLIKE: {metrics_is['QLIKE']:.4f}")
            print(f"    Out-of-sample QLIKE: {metrics_oos['QLIKE']:.4f}")
            print(f"    AIC: {model_gprd.aic:.2f} | BIC: {model_gprd.bic:.2f}")

            # Check if GPRD coefficient is significant
            gprd_param = None
            gprd_pval = None
            for param_name in model_gprd.params.index:
                if 'gprd' in param_name.lower():
                    gprd_param = model_gprd.params[param_name]
                    gprd_pval = model_gprd.pvalues[param_name]
                    break

            if gprd_param is not None:
                sig_marker = '✓' if gprd_pval < 0.05 else '✗'
                print(f"    GPRD coefficient: {gprd_param:.6f} (p={gprd_pval:.3f}) {sig_marker}")

            results.append({
                'commodity': commodity_name,
                'model': 'GARCH + GPRD',
                'AIC': model_gprd.aic,
                'BIC': model_gprd.bic,
                'LogLik': model_gprd.loglikelihood,
                'QLIKE_in_sample': metrics_is['QLIKE'],
                'QLIKE_out_of_sample': metrics_oos['QLIKE'],
                'n_params': len(model_gprd.params),
                'gprd_coef': gprd_param if gprd_param is not None else np.nan,
                'gprd_pval': gprd_pval if gprd_pval is not None else np.nan
            })

            fitted_models['gprd'] = model_gprd

    # Model 4: GARCH + geo + GPRD
    if X_geo_train is not None and X_gprd_train is not None:
        print(f"\n  [4/4] Fitting GARCH + geo_keyword_hits + GPRD...")
        X_combined_df = pd.DataFrame({
            'geo': X_geo_train,
            'gprd': X_gprd_train
        }, index=returns_train.index)

        model_combined = fit_garch_model(returns_train, external_regressors=X_combined_df,
                                         model_name="Combined")

        if model_combined:
            fitted_vol = model_combined.conditional_volatility
            metrics_is = calculate_metrics(returns_train, fitted_vol)

            # Out-of-sample forecast (simplified - using last 20 points for speed)
            try:
                forecast_vols = []
                for i in range(min(20, len(returns_test))):
                    train_window = returns.iloc[:split_idx + i]
                    X_window = pd.DataFrame({
                        'geo': X_geo.iloc[:split_idx + i],
                        'gprd': X_gprd.iloc[:split_idx + i]
                    }, index=train_window.index)
                    model_temp = fit_garch_model(train_window, external_regressors=X_window,
                                                 model_name="Combined_rolling")
                    if model_temp:
                        forecast = model_temp.forecast(horizon=1, reindex=False)
                        forecast_vols.append(np.sqrt(forecast.variance.values[-1, 0]))
                    else:
                        forecast_vols.append(np.nan)

                forecast_vol_series = pd.Series(forecast_vols,
                                                index=returns_test.index[:len(forecast_vols)])
                metrics_oos = calculate_metrics(returns_test.iloc[:len(forecast_vols)],
                                                forecast_vol_series)
            except:
                metrics_oos = {'MSE': np.nan, 'LogLik': np.nan, 'QLIKE': np.nan}

            print(f"    In-sample QLIKE: {metrics_is['QLIKE']:.4f}")
            print(f"    Out-of-sample QLIKE: {metrics_oos['QLIKE']:.4f}")
            print(f"    AIC: {model_combined.aic:.2f} | BIC: {model_combined.bic:.2f}")

            results.append({
                'commodity': commodity_name,
                'model': 'GARCH + geo + GPRD',
                'AIC': model_combined.aic,
                'BIC': model_combined.bic,
                'LogLik': model_combined.loglikelihood,
                'QLIKE_in_sample': metrics_is['QLIKE'],
                'QLIKE_out_of_sample': metrics_oos['QLIKE'],
                'n_params': len(model_combined.params)
            })

            fitted_models['combined'] = model_combined

    return {
        'results': results,
        'models': fitted_models,
        'returns': returns,
        'returns_train': returns_train,
        'returns_test': returns_test
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_garch_comparison(commodity, analysis_results):
    """
    Create visualization comparing GARCH models
    """
    models = analysis_results['models']
    returns_train = analysis_results['returns_train']

    if len(models) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{commodity} - GARCH Model Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Conditional Volatility Comparison
    ax = axes[0, 0]
    for model_name, model in models.items():
        vol = model.conditional_volatility
        ax.plot(vol.index, vol.values, label=model_name.upper(), alpha=0.7, linewidth=1.5)

    ax.set_title('Conditional Volatility Estimates', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Standardized Residuals
    ax = axes[0, 1]
    if 'baseline' in models:
        std_resid = models['baseline'].std_resid
        ax.hist(std_resid.dropna(), bins=50, alpha=0.7, density=True, label='Std. Residuals')

        # Overlay normal distribution
        x = np.linspace(std_resid.min(), std_resid.max(), 100)
        ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')

        ax.set_title('Standardized Residuals Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Standardized Residual')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 3: Squared Returns vs Fitted Variance
    ax = axes[1, 0]
    if 'baseline' in models:
        squared_returns = returns_train ** 2
        fitted_var = models['baseline'].conditional_volatility ** 2

        ax.scatter(fitted_var, squared_returns, alpha=0.3, s=10)
        ax.plot([fitted_var.min(), fitted_var.max()],
                [fitted_var.min(), fitted_var.max()],
                'r--', linewidth=2, label='45° line')

        ax.set_title('Realized vs Fitted Variance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fitted Variance')
        ax.set_ylabel('Squared Returns (Realized Variance)')
        ax.legend()
        ax.grid(alpha=0.3)

    # Plot 4: Model Comparison Metrics
    ax = axes[1, 1]
    results_df = pd.DataFrame(analysis_results['results'])

    if len(results_df) > 0:
        # Plot AIC comparison
        x = np.arange(len(results_df))
        width = 0.35

        ax.bar(x - width / 2, results_df['AIC'], width, label='AIC', alpha=0.8)
        ax.bar(x + width / 2, results_df['BIC'], width, label='BIC', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('GARCH + ', '').replace('GARCH(1,1)', 'Base')
                            for m in results_df['model']], rotation=45, ha='right')
        ax.set_title('Information Criteria Comparison\n(Lower is Better)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(GARCH_RESULTS_DIR, f'{commodity}_garch_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\nLoading enriched data...")
    merged_data = {}

    for name in commodities:
        fname = f"{name.lower()}_enriched.csv"
        path = os.path.join(ENRICHED_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            merged_data[name] = df
            print(f"  ✓ {name}: {len(df)} rows")
        else:
            print(f"  ✗ Missing: {name}")

    all_results = []

    for commodity in merged_data.keys():
        analysis = run_garch_analysis(commodity, merged_data[commodity])

        if analysis and len(analysis['results']) > 0:
            all_results.extend(analysis['results'])
            plot_garch_comparison(commodity, analysis)

    # Save results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(GARCH_RESULTS_DIR, 'garch_comparison_results.csv'),
                          index=False)

        print("\n" + "=" * 80)
        print("SUMMARY: GARCH Model Comparison")
        print("=" * 80)

        display_cols = ['commodity', 'model', 'AIC', 'BIC', 'QLIKE_in_sample',
                        'QLIKE_out_of_sample']
        print(results_df[display_cols].to_string(index=False))

        # Find best model per commodity
        print("\n" + "=" * 80)
        print("Best Model by Commodity (by AIC)")
        print("=" * 80)

        best_models = results_df.loc[results_df.groupby('commodity')['AIC'].idxmin()]
        print(best_models[['commodity', 'model', 'AIC']].to_string(index=False))

        # Check if geopolitical features are significant
        print("\n" + "=" * 80)
        print("Geopolitical Feature Significance")
        print("=" * 80)

        for _, row in results_df.iterrows():
            if 'geo_pval' in row and not pd.isna(row['geo_pval']):
                sig = '✓ SIGNIFICANT' if row['geo_pval'] < 0.05 else '✗ Not significant'
                print(f"{row['commodity']:12s} | geo_keyword_hits: {sig} (p={row['geo_pval']:.3f})")
            if 'gprd_pval' in row and not pd.isna(row['gprd_pval']):
                sig = '✓ SIGNIFICANT' if row['gprd_pval'] < 0.05 else '✗ Not significant'
                print(f"{row['commodity']:12s} | GPRD: {sig} (p={row['gprd_pval']:.3f})")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"Results saved to: {GARCH_RESULTS_DIR}")
        print("  - garch_comparison_results.csv")
        print("  - [Commodity]_garch_comparison.png (visualizations)")
        print("=" * 80)

    else:
        print("\n✗ No successful GARCH models fitted!")


if __name__ == "__main__":
    main()
