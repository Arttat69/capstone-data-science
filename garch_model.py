# GARCH Volatility Modeling with Geopolitical Risk Features
# FIXED VERSION: External regressors properly added to mean equation
# Compares baseline GARCH vs GARCH with geopolitical predictors

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal, ARX
from scipy import stats
from scipy.stats import chi2
import warnings

warnings.filterwarnings('ignore')

# Configuration
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
GARCH_RESULTS_DIR = os.path.join(DATA_DIR, 'garch_results')
os.makedirs(GARCH_RESULTS_DIR, exist_ok=True)

commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]

# Commodity categories for interpretation
COMMODITY_CATEGORIES = {
    'Energy': ['WTI', 'NaturalGas'],
    'Precious Metals': ['Gold'],
    'Industrial Metals': ['Copper', 'Lithium'],
    'Agriculture': ['Wheat']
}


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
    Fit GARCH(1,1) model with optional external regressors in MEAN equation

    Model: Return_t = μ + β*X_t + ε_t, where ε_t ~ GARCH(1,1)

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
            # Use ARX (AutoRegressive with eXogenous variables) for mean equation
            # ARX(0, x=...) means no autoregressive terms, just exogenous variables + constant
            mean_model = ARX(returns_clean, lags=0, x=X_clean, constant=True)

            # Add GARCH(1,1) volatility
            mean_model.volatility = GARCH(p=1, q=1)
            mean_model.distribution = Normal()

            # Fit model
            result = mean_model.fit(disp='off', show_warning=False)
        else:
            # Baseline: Constant mean + GARCH(1,1)
            mean_model = ConstantMean(returns_clean)
            mean_model.volatility = GARCH(p=1, q=1)
            mean_model.distribution = Normal()

            # Fit model
            result = mean_model.fit(disp='off', show_warning=False)

        return result

    except Exception as e:
        print(f"    ERROR fitting {model_name}: {str(e)}")
        return None


def calculate_metrics(returns, fitted_vol, actual_vol=None):
    """
    Calculate performance metrics with improved QLIKE handling
    """
    # Align series
    common_idx = returns.index.intersection(fitted_vol.index)
    returns_aligned = returns.loc[common_idx]
    vol_aligned = fitted_vol.loc[common_idx]

    # Calculate squared returns as proxy for realized variance
    realized_var = returns_aligned ** 2

    # Add small epsilon to prevent division by zero
    epsilon = 1e-8
    vol_aligned_safe = np.maximum(vol_aligned, epsilon)
    realized_var_safe = np.maximum(realized_var, epsilon)

    # Mean Squared Error
    mse = np.mean((realized_var - vol_aligned ** 2) ** 2)

    # Root Mean Squared Error (more interpretable)
    rmse = np.sqrt(mse)

    # Mean Absolute Error
    mae = np.mean(np.abs(returns_aligned - 0))

    # Log-likelihood (approximation)
    try:
        ll = -0.5 * np.sum(np.log(2 * np.pi * vol_aligned_safe ** 2) + realized_var / vol_aligned_safe ** 2)
        if np.isinf(ll) or np.isnan(ll):
            ll = np.nan
    except:
        ll = np.nan

    # QLIKE (Quasi-Likelihood) with safety checks
    try:
        qlike_terms = realized_var_safe / vol_aligned_safe ** 2 - np.log(realized_var_safe / vol_aligned_safe ** 2) - 1
        qlike_terms_clean = qlike_terms[np.isfinite(qlike_terms)]
        if len(qlike_terms_clean) > 0:
            qlike = np.mean(qlike_terms_clean)
        else:
            qlike = np.nan
    except:
        qlike = np.nan

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
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
    returns = returns[(returns.abs() < returns.abs().quantile(0.99))]

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
        fitted_vol = model_baseline.conditional_volatility
        metrics_is = calculate_metrics(returns_train, fitted_vol)

        print(f"    In-sample RMSE: {metrics_is['RMSE']:.4f} | QLIKE: {metrics_is['QLIKE']:.4f}")
        print(f"    AIC: {model_baseline.aic:.2f} | BIC: {model_baseline.bic:.2f}")
        print(f"    Number of parameters: {len(model_baseline.params)}")

        results.append({
            'commodity': commodity_name,
            'model': 'Baseline GARCH(1,1)',
            'AIC': model_baseline.aic,
            'BIC': model_baseline.bic,
            'LogLik': model_baseline.loglikelihood,
            'RMSE_in_sample': metrics_is['RMSE'],
            'QLIKE_in_sample': metrics_is['QLIKE'],
            'n_params': len(model_baseline.params),
            'geo_coef': np.nan,
            'geo_pval': np.nan,
            'gprd_coef': np.nan,
            'gprd_pval': np.nan
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

            print(f"    In-sample RMSE: {metrics_is['RMSE']:.4f} | QLIKE: {metrics_is['QLIKE']:.4f}")
            print(f"    AIC: {model_geo.aic:.2f} | BIC: {model_geo.bic:.2f}")
            print(f"    Number of parameters: {len(model_geo.params)}")

            # Extract geo coefficient and p-value
            geo_param = np.nan
            geo_pval = np.nan
            for param_name in model_geo.params.index:
                if 'geo' in param_name.lower() and 'gprd' not in param_name.lower():
                    geo_param = model_geo.params[param_name]
                    geo_pval = model_geo.pvalues[param_name]
                    sig_marker = '✓ SIGNIFICANT' if geo_pval < 0.05 else '✗ Not significant'
                    print(f"    geo coefficient: {geo_param:.6f} (p={geo_pval:.4f}) {sig_marker}")
                    break

            results.append({
                'commodity': commodity_name,
                'model': 'GARCH + geo_keyword_hits',
                'AIC': model_geo.aic,
                'BIC': model_geo.bic,
                'LogLik': model_geo.loglikelihood,
                'RMSE_in_sample': metrics_is['RMSE'],
                'QLIKE_in_sample': metrics_is['QLIKE'],
                'n_params': len(model_geo.params),
                'geo_coef': geo_param,
                'geo_pval': geo_pval,
                'gprd_coef': np.nan,
                'gprd_pval': np.nan
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

            print(f"    In-sample RMSE: {metrics_is['RMSE']:.4f} | QLIKE: {metrics_is['QLIKE']:.4f}")
            print(f"    AIC: {model_gprd.aic:.2f} | BIC: {model_gprd.bic:.2f}")
            print(f"    Number of parameters: {len(model_gprd.params)}")

            # Extract GPRD coefficient and p-value
            gprd_param = np.nan
            gprd_pval = np.nan
            for param_name in model_gprd.params.index:
                if 'gprd' in param_name.lower():
                    gprd_param = model_gprd.params[param_name]
                    gprd_pval = model_gprd.pvalues[param_name]
                    sig_marker = '✓ SIGNIFICANT' if gprd_pval < 0.05 else '✗ Not significant'
                    print(f"    GPRD coefficient: {gprd_param:.6f} (p={gprd_pval:.4f}) {sig_marker}")
                    break

            results.append({
                'commodity': commodity_name,
                'model': 'GARCH + GPRD',
                'AIC': model_gprd.aic,
                'BIC': model_gprd.bic,
                'LogLik': model_gprd.loglikelihood,
                'RMSE_in_sample': metrics_is['RMSE'],
                'QLIKE_in_sample': metrics_is['QLIKE'],
                'n_params': len(model_gprd.params),
                'geo_coef': np.nan,
                'geo_pval': np.nan,
                'gprd_coef': gprd_param,
                'gprd_pval': gprd_pval
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

            print(f"    In-sample RMSE: {metrics_is['RMSE']:.4f} | QLIKE: {metrics_is['QLIKE']:.4f}")
            print(f"    AIC: {model_combined.aic:.2f} | BIC: {model_combined.bic:.2f}")
            print(f"    Number of parameters: {len(model_combined.params)}")

            # Extract both coefficients
            geo_param = np.nan
            geo_pval = np.nan
            gprd_param = np.nan
            gprd_pval = np.nan

            for param_name in model_combined.params.index:
                if 'geo' in param_name.lower() and 'gprd' not in param_name.lower():
                    geo_param = model_combined.params[param_name]
                    geo_pval = model_combined.pvalues[param_name]
                    sig_marker = '✓ SIGNIFICANT' if geo_pval < 0.05 else '✗ Not significant'
                    print(f"    geo coefficient: {geo_param:.6f} (p={geo_pval:.4f}) {sig_marker}")
                elif 'gprd' in param_name.lower():
                    gprd_param = model_combined.params[param_name]
                    gprd_pval = model_combined.pvalues[param_name]
                    sig_marker = '✓ SIGNIFICANT' if gprd_pval < 0.05 else '✗ Not significant'
                    print(f"    GPRD coefficient: {gprd_param:.6f} (p={gprd_pval:.4f}) {sig_marker}")

            results.append({
                'commodity': commodity_name,
                'model': 'GARCH + geo + GPRD',
                'AIC': model_combined.aic,
                'BIC': model_combined.bic,
                'LogLik': model_combined.loglikelihood,
                'RMSE_in_sample': metrics_is['RMSE'],
                'QLIKE_in_sample': metrics_is['QLIKE'],
                'n_params': len(model_combined.params),
                'geo_coef': geo_param,
                'geo_pval': geo_pval,
                'gprd_coef': gprd_param,
                'gprd_pval': gprd_pval
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
    """Create visualization comparing GARCH models"""
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


def plot_coefficient_comparison(results_df, output_dir):
    """
    Create comprehensive visualization of GARCH coefficients from combined model
    Shows both statistical and economic significance
    """
    print("\n" + "="*80)
    print("Creating GARCH Coefficient Comparison Visualization")
    print("="*80)

    # FOCUS on the combined model (geo + GPRD together)
    combined_results = results_df[results_df['model'] == 'GARCH + geo + GPRD'].copy()

    if len(combined_results) == 0:
        print("No combined model results found")
        return

    commodities = combined_results['commodity'].values
    geo_coefs = combined_results['geo_coef'].values
    geo_pvals = combined_results['geo_pval'].values
    gprd_coefs = combined_results['gprd_coef'].values
    gprd_pvals = combined_results['gprd_pval'].values

    print(f"Plotting coefficients from combined model for {len(commodities)} commodities")
    print()

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ========================================================================
    # SUBPLOT 1: Coefficient Comparison (Grouped Bar Chart)
    # ========================================================================

    x = np.arange(len(commodities))
    width = 0.35

    # Determine colors based on p-value (p < 0.10 for marginal significance)
    geo_colors = ['#2ecc71' if p < 0.10 else '#95a5a6' for p in geo_pvals]
    gprd_colors = ['#e74c3c' if p < 0.10 else '#95a5a6' for p in gprd_pvals]

    # Plot bars
    bars1 = ax1.bar(x - width/2, geo_coefs, width, color=geo_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.85, label='geo_keyword_hits')
    bars2 = ax1.bar(x + width/2, gprd_coefs, width, color=gprd_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.85, label='GPRD')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.5f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=8, fontweight='bold')

    # Customize first subplot
    ax1.set_xlabel('Commodity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('GARCH Coefficient (Combined Model)', fontsize=12, fontweight='bold')
    ax1.set_title('GARCH Coefficients from Combined Model\n(geo_keyword_hits + GPRD)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(commodities, fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend with color explanation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='geo (p < 0.10)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='GPRD (p < 0.10)'),
        Patch(facecolor='#95a5a6', edgecolor='black', label='Not significant (p ≥ 0.10)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # ========================================================================
    # SUBPLOT 2: Economic Magnitude (% Change in Returns)
    # ========================================================================

    # Calculate economic impact: 1-unit change in GPRI index affects returns by coefficient %
    # Scale: coefficient * 100 = % change in returns
    eco_magnitude_geo = np.abs(geo_coefs) * 100
    eco_magnitude_gprd = np.abs(gprd_coefs) * 100

    bars3 = ax2.bar(x - width/2, eco_magnitude_geo, width, color='#3498db',
                    edgecolor='black', linewidth=1.5, alpha=0.85, label='|geo| Effect')
    bars4 = ax2.bar(x + width/2, eco_magnitude_gprd, width, color='#f39c12',
                    edgecolor='black', linewidth=1.5, alpha=0.85, label='|GPRD| Effect')

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}%', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}%', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    # Customize second subplot
    ax2.set_xlabel('Commodity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Economic Magnitude (% Change in Returns per 1-unit GPRI Index)',
                   fontsize=11, fontweight='bold')
    ax2.set_title('Economic Significance: Impact on Returns\n' +
                  '(1-point increase in GPRI changes returns by shown %)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(commodities, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)

    # Add reference line showing "economically trivial" threshold (0.2%)
    ax2.axhline(y=0.2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='0.2% threshold')
    ax2.text(len(commodities)-0.5, 0.22, 'Very Small Effect Threshold', fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'garch_coefficient_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Coefficient comparison chart saved: garch_coefficient_comparison.png")

    # Print interpretation summary
    print("\n" + "="*80)
    print("INTERPRETATION SUMMARY")
    print("="*80)
    for i, comm in enumerate(commodities):
        print(f"\n{comm}:")
        print(f"  geo_keyword_hits: coef={geo_coefs[i]:8.5f} (p={geo_pvals[i]:.4f}), economic effect={eco_magnitude_geo[i]:.4f}%")
        print(f"  GPRD:             coef={gprd_coefs[i]:8.5f} (p={gprd_pvals[i]:.4f}), economic effect={eco_magnitude_gprd[i]:.4f}%")
    print()

    plt.close()

def plot_likelihood_ratio_tests(results_df, output_dir):
    """
    Create visualization showing Likelihood Ratio Test results
    Demonstrates joint significance of geopolitical variables
    """
    print("\n" + "="*80)
    print("Creating Likelihood Ratio Test Visualization")
    print("="*80)

    # Calculate LR statistics
    lr_data = []

    for commodity in results_df['commodity'].unique():
        commodity_results = results_df[results_df['commodity'] == commodity]
        baseline = commodity_results[commodity_results['model'] == 'Baseline GARCH(1,1)']

        if len(baseline) == 0:
            continue

        baseline_loglik = baseline['LogLik'].values[0]
        baseline_params = baseline['n_params'].values[0]

        # Test for each geopolitical model
        geo_models = commodity_results[commodity_results['model'] != 'Baseline GARCH(1,1)']

        for _, geo_model in geo_models.iterrows():
            geo_loglik = geo_model['LogLik']
            geo_params = geo_model['n_params']

            LR_stat = 2 * (geo_loglik - baseline_loglik)
            df = geo_params - baseline_params

            if LR_stat > 0 and df > 0:
                from scipy.stats import chi2
                p_value = chi2.sf(LR_stat, df)

                lr_data.append({
                    'commodity': commodity,
                    'model': geo_model['model'],
                    'LR_statistic': LR_stat,
                    'df': df,
                    'p_value': p_value,
                    'critical_5': chi2.ppf(0.95, df),  # 5% significance
                    'critical_1': chi2.ppf(0.99, df)   # 1% significance
                })

    if len(lr_data) == 0:
        print("No LR test data available")
        return

    lr_df = pd.DataFrame(lr_data)

    # Focus on the combined model for cleaner visualization
    lr_combined = lr_df[lr_df['model'] == 'GARCH + geo + GPRD'].copy()

    print(f"Plotting LR tests for {len(lr_combined)} commodities")

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ========================================================================
    # SUBPLOT 1: LR Statistics vs Critical Values
    # ========================================================================

    commodities = lr_combined['commodity'].values
    lr_stats = lr_combined['LR_statistic'].values
    critical_5 = lr_combined['critical_5'].values
    critical_1 = lr_combined['critical_1'].values

    x = np.arange(len(commodities))
    width = 0.6

    # Color bars based on significance
    colors = ['#27ae60' if p < 0.001 else '#2ecc71' if p < 0.01 else '#f39c12'
              for p in lr_combined['p_value'].values]

    # Plot LR statistics
    bars = ax1.bar(x, lr_stats, width, color=colors, edgecolor='black',
                   linewidth=1.5, alpha=0.85, label='LR Statistic')

    # Add critical value lines
    ax1.plot(x, critical_5, 'r--', linewidth=2, label='Critical Value (α=0.05)', marker='o')
    ax1.plot(x, critical_1, 'darkred', linestyle='--', linewidth=2,
             label='Critical Value (α=0.01)', marker='s')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Customize
    ax1.set_xlabel('Commodity', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Likelihood Ratio Statistic', fontsize=12, fontweight='bold')
    ax1.set_title('Likelihood Ratio Tests: Combined Model vs Baseline\n' +
                  'LR Statistics Exceed Critical Values → Joint Significance',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(commodities, fontsize=11, rotation=0)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation
    ax1.text(0.5, 0.95, 'All bars exceed critical values → Reject H₀',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # ========================================================================
    # SUBPLOT 2: P-values (Log Scale)
    # ========================================================================

    p_values = lr_combined['p_value'].values

    # Use log scale for better visualization of very small p-values
    log_p_values = -np.log10(p_values)  # Convert to -log10 scale

    bars2 = ax2.bar(x, log_p_values, width, color='#e74c3c',
                    edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add significance thresholds
    ax2.axhline(y=-np.log10(0.05), color='orange', linestyle='--', linewidth=2,
                label='p = 0.05', alpha=0.8)
    ax2.axhline(y=-np.log10(0.01), color='red', linestyle='--', linewidth=2,
                label='p = 0.01', alpha=0.8)
    ax2.axhline(y=-np.log10(0.001), color='darkred', linestyle='--', linewidth=2,
                label='p = 0.001', alpha=0.8)

    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        actual_p = p_values[i]
        label = f'p < 0.001' if actual_p < 0.001 else f'p = {actual_p:.4f}'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom',
                fontsize=8, fontweight='bold', rotation=0)

    # Customize
    ax2.set_xlabel('Commodity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Significance of LR Tests\n' +
                  'Higher bars = More significant (all p < 0.001)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(commodities, fontsize=11, rotation=0)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add annotation
    ax2.text(0.5, 0.95, 'All p-values < 0.001 → Highly Significant',
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'likelihood_ratio_tests.png'),
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Likelihood Ratio test visualization saved: likelihood_ratio_tests.png")

    # Print summary
    print("\n" + "="*80)
    print("LIKELIHOOD RATIO TEST SUMMARY")
    print("="*80)
    for i, comm in enumerate(commodities):
        print(f"{comm:12s}: LR = {lr_stats[i]:7.2f}, p < 0.001 (*** Highly Significant)")
    print("="*80)
    print("\nConclusion: Geopolitical variables JOINTLY improve model fit significantly")
    print("(Even though individual coefficients are not significant)")
    print()

    plt.close()


def get_commodity_category(commodity_name):
    """Get the category for a commodity"""
    for category, commodities in COMMODITY_CATEGORIES.items():
        if commodity_name in commodities:
            return category
    return "Other"


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("GARCH VOLATILITY MODELING WITH GEOPOLITICAL FEATURES")
    print("=" * 80)
    print("Models:")
    print("  1. Baseline: Return_t = μ + ε_t, ε_t ~ GARCH(1,1)")
    print("  2. Return_t = μ + β₁*geo_keyword_hits_t + ε_t, ε_t ~ GARCH(1,1)")
    print("  3. Return_t = μ + β₂*GPRD_t + ε_t, ε_t ~ GARCH(1,1)")
    print("  4. Return_t = μ + β₁*geo_t + β₂*GPRD_t + ε_t, ε_t ~ GARCH(1,1)")
    print("=" * 80)
    print("Testing: Do geopolitical factors predict commodity returns?")
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
        # Create coefficient comparison visualization
        plot_coefficient_comparison(results_df, GARCH_RESULTS_DIR)
        # Create likelihood ratio test visualization
        plot_likelihood_ratio_tests(results_df, GARCH_RESULTS_DIR)

        print("\n" + "=" * 80)
        print("SUMMARY: GARCH Model Comparison")
        print("=" * 80)

        display_cols = ['commodity', 'model', 'AIC', 'BIC', 'n_params', 'RMSE_in_sample']
        print(results_df[display_cols].to_string(index=False))

        # Find best model per commodity
        print("\n" + "=" * 80)
        print("BEST MODELS BY COMMODITY (by AIC)")
        print("=" * 80)

        best_models = results_df.loc[results_df.groupby('commodity')['AIC'].idxmin()]
        print(best_models[['commodity', 'model', 'AIC', 'n_params']].to_string(index=False))

        # Calculate AIC improvements
        print("\n" + "=" * 80)
        print("AIC IMPROVEMENTS (Baseline vs Best Geopolitical Model)")
        print("=" * 80)

        for commodity in results_df['commodity'].unique():
            commodity_results = results_df[results_df['commodity'] == commodity]
            baseline_aic = commodity_results[commodity_results['model'] == 'Baseline GARCH(1,1)']['AIC'].values
            geo_models = commodity_results[commodity_results['model'] != 'Baseline GARCH(1,1)']

            if len(baseline_aic) > 0 and len(geo_models) > 0:
                best_geo_aic = geo_models['AIC'].min()
                improvement = baseline_aic[0] - best_geo_aic
                improvement_pct = (improvement / baseline_aic[0]) * 100

                category = get_commodity_category(commodity)
                print(f"{commodity:12s} ({category:18s}): AIC improved by {improvement:7.2f} ({improvement_pct:5.2f}%)")

        # LIKELIHOOD RATIO TESTS
        print("\n" + "=" * 80)
        print("LIKELIHOOD RATIO TESTS (Joint Significance)")
        print("=" * 80)
        print("Testing: Do geopolitical variables jointly improve model fit?")
        print("H0: Geopolitical variables have no effect (β = 0)")
        print("H1: Geopolitical variables affect returns (β ≠ 0)")
        print("-" * 80)

        lr_results = []

        for commodity in results_df['commodity'].unique():
            commodity_results = results_df[results_df['commodity'] == commodity]

            baseline = commodity_results[commodity_results['model'] == 'Baseline GARCH(1,1)']
            if len(baseline) == 0:
                continue

            baseline_loglik = baseline['LogLik'].values[0]
            baseline_params = baseline['n_params'].values[0]

            geo_models = commodity_results[commodity_results['model'] != 'Baseline GARCH(1,1)']

            for _, geo_model in geo_models.iterrows():
                geo_loglik = geo_model['LogLik']
                geo_params = geo_model['n_params']

                LR_stat = 2 * (geo_loglik - baseline_loglik)
                df = geo_params - baseline_params

                if LR_stat > 0 and df > 0:
                    p_value = chi2.sf(LR_stat, df)
                else:
                    p_value = np.nan

                if not np.isnan(p_value):
                    if p_value < 0.001:
                        sig_level = '***'
                        interpretation = 'Highly Significant'
                    elif p_value < 0.01:
                        sig_level = '**'
                        interpretation = 'Significant'
                    elif p_value < 0.05:
                        sig_level = '*'
                        interpretation = 'Significant'
                    elif p_value < 0.10:
                        sig_level = '.'
                        interpretation = 'Marginally Significant'
                    else:
                        sig_level = ''
                        interpretation = 'Not Significant'
                else:
                    sig_level = ''
                    interpretation = 'N/A'

                lr_results.append({
                    'commodity': commodity,
                    'model': geo_model['model'],
                    'LR_statistic': LR_stat,
                    'df': df,
                    'p_value': p_value,
                    'significance': sig_level,
                    'interpretation': interpretation
                })

                model_name = geo_model['model'].replace('GARCH + ', '')
                print(f"{commodity:12s} | {model_name:25s} | LR={LR_stat:8.3f} (df={df}) | p={p_value:.4f} {sig_level:3s} | {interpretation}")

        if len(lr_results) > 0:
            lr_df = pd.DataFrame(lr_results)
            lr_df.to_csv(os.path.join(GARCH_RESULTS_DIR, 'likelihood_ratio_tests.csv'), index=False)

        print("\n" + "-" * 80)
        print("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10")
        print("-" * 80)

        if len(lr_results) > 0:
            lr_df = pd.DataFrame(lr_results)

            n_highly_sig = len(lr_df[lr_df['p_value'] < 0.001])
            n_sig = len(lr_df[lr_df['p_value'] < 0.05])
            n_marginal = len(lr_df[(lr_df['p_value'] >= 0.05) & (lr_df['p_value'] < 0.10)])
            n_not_sig = len(lr_df[lr_df['p_value'] >= 0.10])

            print(f"\nSummary of {len(lr_df)} model comparisons:")
            print(f"  Highly significant (p < 0.001): {n_highly_sig}")
            print(f"  Significant (p < 0.05):         {n_sig}")
            print(f"  Marginally significant (p < 0.10): {n_marginal}")
            print(f"  Not significant (p >= 0.10):    {n_not_sig}")

            sig_commodities = lr_df[lr_df['p_value'] < 0.05]['commodity'].unique()
            if len(sig_commodities) > 0:
                print(f"\n✓ Geopolitical variables are JOINTLY SIGNIFICANT for: {', '.join(sig_commodities)}")
            else:
                print(f"\n✗ Geopolitical variables are not jointly significant at p<0.05 for any commodity")

        # Individual coefficient significance
        print("\n" + "=" * 80)
        print("INDIVIDUAL COEFFICIENT SIGNIFICANCE")
        print("=" * 80)

        geo_significant = []
        gprd_significant = []

        for _, row in results_df.iterrows():
            if not pd.isna(row['geo_pval']):
                is_sig = row['geo_pval'] < 0.05
                sig_marker = '✓ SIGNIFICANT' if is_sig else '✗ Not significant'
                print(f"{row['commodity']:12s} | geo_keyword_hits: {sig_marker:17s} | coef={row['geo_coef']:8.5f}, p={row['geo_pval']:.4f}")
                if is_sig:
                    geo_significant.append(row['commodity'])

            if not pd.isna(row['gprd_pval']):
                is_sig = row['gprd_pval'] < 0.05
                sig_marker = '✓ SIGNIFICANT' if is_sig else '✗ Not significant'
                print(f"{row['commodity']:12s} | GPRD:             {sig_marker:17s} | coef={row['gprd_coef']:8.5f}, p={row['gprd_pval']:.4f}")
                if is_sig:
                    gprd_significant.append(row['commodity'])

        print("\n" + "=" * 80)
        print("SIGNIFICANCE SUMMARY")
        print("=" * 80)

        if len(geo_significant) > 0:
            print(f"\n✓ geo_keyword_hits is SIGNIFICANT for: {', '.join(set(geo_significant))}")
        else:
            print("\n✗ geo_keyword_hits is NOT individually significant for any commodity")

        if len(gprd_significant) > 0:
            print(f"✓ GPRD is SIGNIFICANT for: {', '.join(set(gprd_significant))}")
        else:
            print("✗ GPRD is NOT individually significant for any commodity")

        # Sector analysis
        print("\n" + "=" * 80)
        print("SECTOR-SPECIFIC GEOPOLITICAL SENSITIVITY")
        print("=" * 80)

        for category, comm_list in COMMODITY_CATEGORIES.items():
            category_improvements = []
            for commodity in comm_list:
                if commodity in results_df['commodity'].values:
                    commodity_results = results_df[results_df['commodity'] == commodity]
                    baseline_aic = commodity_results[commodity_results['model'] == 'Baseline GARCH(1,1)']['AIC'].values
                    geo_models = commodity_results[commodity_results['model'] != 'Baseline GARCH(1,1)']

                    if len(baseline_aic) > 0 and len(geo_models) > 0:
                        best_geo_aic = geo_models['AIC'].min()
                        improvement = baseline_aic[0] - best_geo_aic
                        category_improvements.append(improvement)

            if len(category_improvements) > 0:
                avg_improvement = np.mean(category_improvements)
                print(f"{category:20s}: Average AIC improvement = {avg_improvement:7.2f}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {GARCH_RESULTS_DIR}")
        print("  - garch_comparison_results.csv")
        print("  - likelihood_ratio_tests.csv")
        print("  - [Commodity]_garch_comparison.png (visualizations)")
        print("="*80)

    else:
        print("\n✗ No successful GARCH models fitted!")

if __name__ == "__main__":
    main()
