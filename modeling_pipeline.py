# modeling_pipeline.py
# NOW PREDICTS BOTH RETURNS AND VOLATILITY

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, roc_auc_score)
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
ENRICHED_DIR = os.path.join(DATA_DIR, "enriched")
MODEL_RESULTS_DIR_MOD = os.path.join(DATA_DIR, 'model_results_modelling')
os.makedirs(MODEL_RESULTS_DIR_MOD, exist_ok=True)

commodities = ["Gold", "WTI", "Wheat", "NaturalGas", "Copper", "Lithium"]

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


def add_lagged_features(df, feature_cols, max_lag=5):
    """Add lagged features to ensure no lookahead bias"""
    for col in feature_cols:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df


def prepare_data(df, lag_features):
    """Prepare data with proper handling of missing values and lagged features"""
    # Fill missing values with forward fill first, then 0
    df['geo_keyword_hits'] = df['geo_keyword_hits'].fillna(method='ffill').fillna(0)
    df['sentiment'] = df['sentiment'].fillna(method='ffill').fillna(0)
    if 'EVENT' in df.columns:
        df['EVENT'].fillna('None', inplace=True)

    # Create volatility target if not exists (rolling 5-day realized volatility)
    if 'Vol_5' not in df.columns:
        df['Vol_5'] = df['Return'].rolling(window=5).std()

    # Add lagged volatility for prediction
    df['Vol_5_lag1'] = df['Vol_5'].shift(1)

    # Add lagged features
    df = add_lagged_features(df, lag_features, max_lag=5)

    # Drop rows with NaN in critical features
    df.dropna(inplace=True)

    return df


def create_holdout_split(df, holdout_years=2):
    """Create train/validation + holdout split"""
    df = df.sort_values('Date').reset_index(drop=True)
    holdout_date = df['Date'].max() - pd.DateOffset(years=holdout_years)

    train_val = df[df['Date'] < holdout_date].copy()
    holdout = df[df['Date'] >= holdout_date].copy()

    return train_val, holdout


def dummy_baseline_predict(y_train, X_test):
    """Dummy baseline: predict historical mean"""
    return np.full(len(X_test), y_train.mean())


def zero_baseline_predict(X_test):
    """Zero baseline: predict 0 for returns"""
    return np.zeros(len(X_test))


def evaluate_regression(y_true, y_pred, model_name):
    """Compute multiple regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        f'{model_name}_RMSE': rmse,
        f'{model_name}_MAE': mae,
        f'{model_name}_R2': r2
    }


def walk_forward_regression(df, features, target, n_splits=5):
    """Walk-forward validation for regression models"""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        'fold': [],
        'train_size': [],
        'test_size': []
    }

    # Initialize metric collectors
    models = ['Zero', 'Mean', 'LR_Baseline', 'LR_Enhanced', 'RF_Enhanced']
    for model in models:
        for metric in ['RMSE', 'MAE', 'R2']:
            results[f'{model}_{metric}'] = []

    features_baseline = ['Return_lag1'] if target == 'Return' else ['Vol_5_lag1']
    features_enhanced = features

    X = df[features_enhanced].values
    y = df[target].values

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Extract baseline features
        X_train_base = df.iloc[train_idx][features_baseline].values
        X_test_base = df.iloc[test_idx][features_baseline].values

        results['fold'].append(fold + 1)
        results['train_size'].append(len(train_idx))
        results['test_size'].append(len(test_idx))

        # 1. Zero baseline
        y_pred_zero = zero_baseline_predict(X_test)
        zero_metrics = evaluate_regression(y_test, y_pred_zero, 'Zero')

        # 2. Mean baseline
        y_pred_mean = dummy_baseline_predict(y_train, X_test)
        mean_metrics = evaluate_regression(y_test, y_pred_mean, 'Mean')

        # 3. Baseline LR (only lag1)
        lr_base = LinearRegression()
        lr_base.fit(X_train_base, y_train)
        y_pred_lr_base = lr_base.predict(X_test_base)
        lr_base_metrics = evaluate_regression(y_test, y_pred_lr_base, 'LR_Baseline')

        # 4. Enhanced LR (all features with scaling)
        pipe_lr_enh = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LinearRegression())
        ])
        pipe_lr_enh.fit(X_train, y_train)
        y_pred_lr_enh = pipe_lr_enh.predict(X_test)
        lr_enh_metrics = evaluate_regression(y_test, y_pred_lr_enh, 'LR_Enhanced')

        # 5. Enhanced RF (no scaling needed for trees)
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_metrics = evaluate_regression(y_test, y_pred_rf, 'RF_Enhanced')

        # Store all metrics
        for metrics_dict in [zero_metrics, mean_metrics, lr_base_metrics,
                             lr_enh_metrics, rf_metrics]:
            for key, val in metrics_dict.items():
                results[key].append(val)

    return pd.DataFrame(results)


def walk_forward_classification(df, features, target='Return_binary', n_splits=5):
    """Walk-forward validation for classification"""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        'fold': [],
        'train_size': [],
        'test_size': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'roc_auc': []
    }

    X = df[features].values
    y = df[target].values

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Pipeline with scaling fitted on train only
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=200, random_state=42))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]

        results['fold'].append(fold + 1)
        results['train_size'].append(len(train_idx))
        results['test_size'].append(len(test_idx))
        results['accuracy'].append(accuracy_score(y_test, y_pred))
        results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        results['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))

    return pd.DataFrame(results)


def find_optimal_clusters(df, regime_features, max_k=5):
    """Find optimal number of clusters using silhouette score"""
    # Scale features for KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[regime_features].fillna(0))

    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    optimal_k = K_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores


def fit_regimes_on_train(train_df, test_df, regime_features, n_clusters):
    """Fit clustering on train, predict on test"""
    scaler = StandardScaler()

    # Fit on train
    X_train = scaler.fit_transform(train_df[regime_features].fillna(0))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df['Regime'] = kmeans.fit_predict(X_train)

    # Predict on test
    X_test = scaler.transform(test_df[regime_features].fillna(0))
    test_df['Regime'] = kmeans.predict(X_test)

    return train_df, test_df, kmeans


def evaluate_regimes_economically(df, regime_features):
    """Validate that regimes have different volatility/return characteristics"""
    regime_stats = df.groupby('Regime').agg({
        'Return': ['mean', 'std'],
        'Vol_5': 'mean',
        **{feat: 'mean' for feat in regime_features}
    }).round(4)

    return regime_stats


def main():
    lag_features = ['GPRD', 'geo_keyword_hits']
    target_return = 'Return'
    target_volatility = 'Vol_5'  # NEW: Volatility target

    all_regression_results = []
    all_volatility_results = []  # NEW: Store volatility predictions
    all_classification_results = []
    holdout_results = []
    regime_analysis = []

    for commodity, df in merged_data.items():
        print(f"\n{'=' * 60}")
        print(f"Processing commodity: {commodity}")
        print(f"{'=' * 60}")

        # Prepare data
        df = prepare_data(df, lag_features)
        print(f"{commodity} - {len(df)} rows after preprocessing")

        if len(df) < 100:
            print(f"Insufficient data for {commodity}, skipping.")
            continue

        # Define features for returns
        features_enhanced = ['Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment'] + \
                            [f'{feat}_lag{lag}' for feat in lag_features for lag in range(1, 6)]

        # Define features for volatility (includes lagged volatility)
        features_volatility = ['Vol_5_lag1', 'Return_lag1', 'GPRD', 'geo_keyword_hits', 'sentiment'] + \
                              [f'{feat}_lag{lag}' for feat in lag_features for lag in range(1, 6)]

        # Ensure all features are numeric
        all_features = list(set(features_enhanced + features_volatility))
        df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=all_features + [target_return, target_volatility], inplace=True)

        # Create holdout set
        train_val_df, holdout_df = create_holdout_split(df, holdout_years=2)
        print(f"Train/Val size: {len(train_val_df)}, Holdout size: {len(holdout_df)}")

        if len(train_val_df) < 50:
            print(f"Insufficient train/val data for {commodity}, skipping.")
            continue

        # --- REGRESSION: Walk-forward validation for RETURNS ---
        print(f"\nRunning walk-forward RETURN prediction validation...")
        regression_results = walk_forward_regression(
            train_val_df, features_enhanced, target_return, n_splits=5
        )
        regression_results['Commodity'] = commodity
        regression_results['Target'] = 'Return'
        all_regression_results.append(regression_results)

        # Print summary statistics
        print(f"\nReturn Prediction Results Summary (mean across folds):")
        metric_cols = [c for c in regression_results.columns if '_RMSE' in c or '_MAE' in c or '_R2' in c]
        print(regression_results[metric_cols].mean())

        # --- NEW: REGRESSION FOR VOLATILITY ---
        print(f"\nRunning walk-forward VOLATILITY prediction validation...")
        volatility_results = walk_forward_regression(
            train_val_df, features_volatility, target_volatility, n_splits=5
        )
        volatility_results['Commodity'] = commodity
        volatility_results['Target'] = 'Volatility'
        all_volatility_results.append(volatility_results)

        # Print summary statistics
        print(f"\nVolatility Prediction Results Summary (mean across folds):")
        metric_cols_vol = [c for c in volatility_results.columns if '_RMSE' in c or '_MAE' in c or '_R2' in c]
        print(volatility_results[metric_cols_vol].mean())

        # --- CLASSIFICATION: Walk-forward validation ---
        train_val_df['Return_binary'] = (train_val_df['Return'] > 0).astype(int)
        holdout_df['Return_binary'] = (holdout_df['Return'] > 0).astype(int)

        print(f"\nRunning walk-forward classification validation...")
        classification_results = walk_forward_classification(
            train_val_df, features_enhanced, n_splits=5
        )
        classification_results['Commodity'] = commodity
        all_classification_results.append(classification_results)

        print(f"\nClassification Results Summary (mean across folds):")
        print(classification_results[['accuracy', 'precision', 'recall', 'roc_auc']].mean())

        # --- CLUSTERING: Fit on train, evaluate on both ---
        regime_features = ['Vol_5', 'GPRD', 'geo_keyword_hits']

        # Find optimal clusters
        optimal_k, silhouette_scores = find_optimal_clusters(train_val_df, regime_features, max_k=5)
        print(f"\nOptimal number of clusters: {optimal_k}")

        # Fit regimes
        train_val_df, holdout_df, kmeans = fit_regimes_on_train(
            train_val_df, holdout_df, regime_features, n_clusters=optimal_k
        )

        # Economic validation
        regime_stats = evaluate_regimes_economically(train_val_df, regime_features)
        print(f"\nRegime Statistics (Train/Val):")
        print(regime_stats)

        regime_analysis.append({
            'Commodity': commodity,
            'Optimal_K': optimal_k,
            'Regime_Stats': regime_stats.to_dict()
        })

        # --- HOLDOUT EVALUATION ---
        if len(holdout_df) > 10:
            print(f"\nEvaluating on final holdout set...")

            holdout_result = {'Commodity': commodity, 'Holdout_Size': len(holdout_df)}

            # RETURNS: Regression on holdout
            X_holdout_ret = holdout_df[features_enhanced].values
            y_holdout_ret = holdout_df[target_return].values

            pipe_lr_ret = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
            pipe_lr_ret.fit(train_val_df[features_enhanced], train_val_df[target_return])
            y_pred_lr_ret = pipe_lr_ret.predict(X_holdout_ret)

            rf_final_ret = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                max_features='sqrt', random_state=42
            )
            rf_final_ret.fit(train_val_df[features_enhanced], train_val_df[target_return])
            y_pred_rf_ret = rf_final_ret.predict(X_holdout_ret)

            holdout_result.update({
                **{f'Return_{k}': v for k, v in evaluate_regression(y_holdout_ret, y_pred_lr_ret, 'LR_Enhanced').items()},
                **{f'Return_{k}': v for k, v in evaluate_regression(y_holdout_ret, y_pred_rf_ret, 'RF_Enhanced').items()}
            })

            # VOLATILITY: Regression on holdout
            X_holdout_vol = holdout_df[features_volatility].values
            y_holdout_vol = holdout_df[target_volatility].values

            pipe_lr_vol = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
            pipe_lr_vol.fit(train_val_df[features_volatility], train_val_df[target_volatility])
            y_pred_lr_vol = pipe_lr_vol.predict(X_holdout_vol)

            rf_final_vol = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=5,
                max_features='sqrt', random_state=42
            )
            rf_final_vol.fit(train_val_df[features_volatility], train_val_df[target_volatility])
            y_pred_rf_vol = rf_final_vol.predict(X_holdout_vol)

            holdout_result.update({
                **{f'Volatility_{k}': v for k, v in evaluate_regression(y_holdout_vol, y_pred_lr_vol, 'LR_Enhanced').items()},
                **{f'Volatility_{k}': v for k, v in evaluate_regression(y_holdout_vol, y_pred_rf_vol, 'RF_Enhanced').items()}
            })

            holdout_results.append(holdout_result)

            print(f"Holdout Results (Returns):")
            print(f"  LR RMSE: {[v for k, v in holdout_result.items() if 'Return_LR_Enhanced_RMSE' in k][0]:.6f}")
            print(f"  RF RMSE: {[v for k, v in holdout_result.items() if 'Return_RF_Enhanced_RMSE' in k][0]:.6f}")
            print(f"Holdout Results (Volatility):")
            print(f"  LR RMSE: {[v for k, v in holdout_result.items() if 'Volatility_LR_Enhanced_RMSE' in k][0]:.6f}")
            print(f"  RF RMSE: {[v for k, v in holdout_result.items() if 'Volatility_RF_Enhanced_RMSE' in k][0]:.6f}")

    # --- SAVE RESULTS ---
    if all_regression_results:
        regression_df = pd.concat(all_regression_results, ignore_index=True)
        regression_df.to_csv(
            os.path.join(MODEL_RESULTS_DIR_MOD, 'walk_forward_return_prediction.csv'),
            index=False
        )

    if all_volatility_results:
        volatility_df = pd.concat(all_volatility_results, ignore_index=True)
        volatility_df.to_csv(
            os.path.join(MODEL_RESULTS_DIR_MOD, 'walk_forward_volatility_prediction.csv'),
            index=False
        )

    if all_classification_results:
        classification_df = pd.concat(all_classification_results, ignore_index=True)
        classification_df.to_csv(
            os.path.join(MODEL_RESULTS_DIR_MOD, 'walk_forward_classification_results.csv'),
            index=False
        )

    if holdout_results:
        holdout_df_final = pd.DataFrame(holdout_results)
        holdout_df_final.to_csv(
            os.path.join(MODEL_RESULTS_DIR_MOD, 'holdout_results_complete.csv'),
            index=False
        )

    # --- VISUALIZATION ---
    plot_results(regression_df, volatility_df, classification_df, holdout_df_final)

    print(f"\n{'=' * 60}")
    print("Pipeline complete! Results saved to:", MODEL_RESULTS_DIR_MOD)
    print(f"{'=' * 60}")


def plot_results(regression_df, volatility_df, classification_df, holdout_df):
    """Create comprehensive visualizations for BOTH returns and volatility"""

    # 1. RETURNS: Mean RMSE across models and commodities
    plt.figure(figsize=(14, 8))
    rmse_cols = [c for c in regression_df.columns if '_RMSE' in c]
    rmse_summary = regression_df.groupby('Commodity')[rmse_cols].mean()
    ax = rmse_summary.plot.bar(rot=45, width=0.8)
    plt.title('RETURN Prediction: Mean RMSE Across Walk-Forward Folds\n(Lower is Better)',
              fontsize=14, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12)
    plt.xlabel('Commodity', fontsize=12)
    plt.legend(title='Model', labels=[c.replace('_RMSE', '') for c in rmse_cols],
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_RESULTS_DIR_MOD, 'return_rmse_comparison.png'),
                dpi=300, bbox_inches='tight')

    # 2. VOLATILITY: Mean RMSE across models and commodities
    plt.figure(figsize=(14, 8))
    vol_rmse_cols = [c for c in volatility_df.columns if '_RMSE' in c]
    vol_rmse_summary = volatility_df.groupby('Commodity')[vol_rmse_cols].mean()
    ax = vol_rmse_summary.plot.bar(rot=45, width=0.8, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'])
    plt.title('VOLATILITY Prediction: Mean RMSE Across Walk-Forward Folds\n(Lower is Better)',
              fontsize=14, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12)
    plt.xlabel('Commodity', fontsize=12)
    plt.legend(title='Model', labels=[c.replace('_RMSE', '') for c in vol_rmse_cols],
               bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_RESULTS_DIR_MOD, 'volatility_rmse_comparison.png'),
                dpi=300, bbox_inches='tight')

    # 3. COMPARISON: Returns vs Volatility R² (to show volatility is more predictable)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Returns R²
    r2_cols = [c for c in regression_df.columns if '_R2' in c and 'RF_Enhanced' in c]
    r2_summary = regression_df.groupby('Commodity')[r2_cols].mean()
    r2_summary.plot.bar(ax=axes[0], rot=45, color='steelblue', legend=False)
    axes[0].set_title('Return Prediction R²\n(Random Forest Enhanced)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R² Score', fontsize=10)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_ylim(-0.5, 1.0)

    # Volatility R²
    vol_r2_cols = [c for c in volatility_df.columns if '_R2' in c and 'RF_Enhanced' in c]
    vol_r2_summary = volatility_df.groupby('Commodity')[vol_r2_cols].mean()
    vol_r2_summary.plot.bar(ax=axes[1], rot=45, color='darkorange', legend=False)
    axes[1].set_title('Volatility Prediction R²\n(Random Forest Enhanced)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('R² Score', fontsize=10)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_ylim(-0.5, 1.0)

    plt.suptitle('Comparing Predictability: Returns vs Volatility\n(Higher R² = More Predictable)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_RESULTS_DIR_MOD, 'returns_vs_volatility_r2.png'),
                dpi=300, bbox_inches='tight')

    # 4. Classification metrics (unchanged)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = ['accuracy', 'precision', 'recall', 'roc_auc']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        metric_summary = classification_df.groupby('Commodity')[metric].mean()
        metric_summary.plot.bar(ax=ax, color='steelblue', rot=45)
        ax.set_title(f'Mean {metric.upper()} Across Folds', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_xlabel('Commodity', fontsize=10)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle('Classification Performance: Logistic Regression with Enhanced Features',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_RESULTS_DIR_MOD, 'classification_metrics.png'),
                dpi=300, bbox_inches='tight')

    # 5. Holdout performance for BOTH targets
    if holdout_df is not None and not holdout_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Returns holdout
        return_cols = [c for c in holdout_df.columns if 'Return_' in c and '_RMSE' in c]
        if return_cols:
            holdout_return = holdout_df.set_index('Commodity')[return_cols]
            holdout_return.columns = [c.replace('Return_', '').replace('_RMSE', '') for c in holdout_return.columns]
            holdout_return.plot.bar(ax=axes[0], rot=45, width=0.7)
            axes[0].set_title('RETURN Prediction - Holdout Set (Final 2 Years)', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('RMSE', fontsize=10)
            axes[0].legend(['Linear Regression', 'Random Forest'])

        # Volatility holdout
        vol_cols = [c for c in holdout_df.columns if 'Volatility_' in c and '_RMSE' in c]
        if vol_cols:
            holdout_vol = holdout_df.set_index('Commodity')[vol_cols]
            holdout_vol.columns = [c.replace('Volatility_', '').replace('_RMSE', '') for c in holdout_vol.columns]
            holdout_vol.plot.bar(ax=axes[1], rot=45, width=0.7, color=['darkorange', 'darkred'])
            axes[1].set_title('VOLATILITY Prediction - Holdout Set (Final 2 Years)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('RMSE', fontsize=10)
            axes[1].legend(['Linear Regression', 'Random Forest'])

        plt.suptitle('Final Holdout Performance: Never Seen During Training',
                     fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_RESULTS_DIR_MOD, 'holdout_performance_complete.png'),
                    dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()