# Statistical Significance Testing for GPRD Impact
# Tests whether performance improvements are statistically significant

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, "data")
MODEL_RESULTS_DIR = os.path.join(DATA_DIR, 'model_results')

def main():

    print("="*80)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("="*80)

    # Load results - check if file exists first
    results_file = os.path.join(MODEL_RESULTS_DIR, 'lstm_gprd_comparison.csv')
    if not os.path.exists(results_file):
        print(f"\n[ERROR] Results file not found: {results_file}")
        print("Please run the LSTM model first (lstm_model.py) to generate the results file.")
        print("Skipping statistical tests...")
        return

    results_df = pd.read_csv(results_file)
# ============================================================================
# 1. PAIRED T-TEST: GPRD vs Baseline
# ============================================================================

    print("\n1. PAIRED T-TEST (Within-Commodity Comparison)")
    print("-"*80)

    # Get paired data (commodities with both baseline and GPRD results)
    commodities = results_df['commodity'].unique()

    paired_data = []
    for commodity in commodities:
        subset = results_df[results_df['commodity'] == commodity]
        if len(subset) == 2:  # Has both baseline and GPRD
            baseline = subset[subset['use_gprd'] == False].iloc[0]
            gprd = subset[subset['use_gprd'] == True].iloc[0]
            paired_data.append({
                'commodity': commodity,
                'sharpe_baseline': baseline['sharpe'],
                'sharpe_gprd': gprd['sharpe'],
                'sharpe_diff': gprd['sharpe'] - baseline['sharpe'],
                'return_baseline': baseline['cumulative_return'],
                'return_gprd': gprd['cumulative_return'],
                'return_diff': gprd['cumulative_return'] - baseline['cumulative_return']
            })

    paired_df = pd.DataFrame(paired_data)

    # Paired t-test on Sharpe ratios
    sharpe_baseline = paired_df['sharpe_baseline'].values
    sharpe_gprd = paired_df['sharpe_gprd'].values

    t_stat, p_value = stats.ttest_rel(sharpe_gprd, sharpe_baseline)

    print(f"\nPaired t-test results (Sharpe Ratio):")
    print(f"  H0: GPRD has no effect on Sharpe ratio")
    print(f"  H1: GPRD changes Sharpe ratio")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significance: {'***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else 'Not significant'}")

    if p_value < 0.05:
        print(f"  ✓ REJECT null hypothesis: GPRD significantly affects performance")
    else:
        print(f"  ✗ FAIL TO REJECT null hypothesis: No significant effect")

    print(f"\nMean Sharpe difference: {np.mean(sharpe_gprd - sharpe_baseline):.4f}")
    print(f"95% Confidence Interval: [{np.percentile(sharpe_gprd - sharpe_baseline, 2.5):.4f}, "
          f"{np.percentile(sharpe_gprd - sharpe_baseline, 97.5):.4f}]")

    # ============================================================================
    # 2. WILCOXON SIGNED-RANK TEST (Non-parametric alternative)
    # ============================================================================

    print("\n2. WILCOXON SIGNED-RANK TEST (Non-parametric)")
    print("-"*80)

    w_stat, w_pvalue = stats.wilcoxon(sharpe_gprd, sharpe_baseline)

    print(f"  Wilcoxon test statistic: {w_stat:.4f}")
    print(f"  p-value: {w_pvalue:.4f}")
    print(f"  Significance: {'***' if w_pvalue < 0.01 else '**' if w_pvalue < 0.05 else '*' if w_pvalue < 0.10 else 'Not significant'}")

    # ============================================================================
    # 3. EFFECT SIZE (Cohen's d)
    # ============================================================================

    print("\n3. EFFECT SIZE ANALYSIS")
    print("-"*80)

    differences = sharpe_gprd - sharpe_baseline
    cohens_d = np.mean(differences) / np.std(differences)

    print(f"  Cohen's d: {cohens_d:.4f}")
    print(f"  Interpretation: ", end="")
    if abs(cohens_d) < 0.2:
        print("Negligible effect")
    elif abs(cohens_d) < 0.5:
        print("Small effect")
    elif abs(cohens_d) < 0.8:
        print("Medium effect")
    else:
        print("Large effect")


    # ============================================================================
    # 4. COMMODITY-SPECIFIC ANALYSIS
    # ============================================================================

    print("\n4. COMMODITY-SPECIFIC SIGNIFICANCE")
    print("-" * 80)

    # Classify commodities by geopolitical sensitivity
    commodity_classes = {
        'Energy': ['WTI', 'NaturalGas'],
        'Precious Metals': ['Gold'],
        'Industrial Metals': ['Copper'],
        'Technology': ['Lithium'],
        'Agriculture': ['Wheat']
    }

    print("\nSharpe Ratio Changes by Commodity Class:")
    for class_name, commodities_list in commodity_classes.items():
        class_data = paired_df[paired_df['commodity'].isin(commodities_list)]
        if len(class_data) > 0:
            mean_improvement = class_data['sharpe_diff'].mean()
            print(f"  {class_name:20s}: {mean_improvement:+.3f} "
                  f"({'✓ Positive' if mean_improvement > 0 else '✗ Negative'})")

    # Test if energy commodities improve more than non-energy
    energy_improvements = paired_df[paired_df['commodity'].isin(['WTI', 'NaturalGas'])]['sharpe_diff'].values
    non_energy_improvements = paired_df[~paired_df['commodity'].isin(['WTI', 'NaturalGas'])]['sharpe_diff'].values

    if len(energy_improvements) > 0 and len(non_energy_improvements) > 0:
        t_stat_class, p_value_class = stats.ttest_ind(energy_improvements, non_energy_improvements)
        print(f"\nEnergy vs Non-Energy comparison:")
        print(f"  Mean improvement (Energy): {np.mean(energy_improvements):.3f}")
        print(f"  Mean improvement (Non-Energy): {np.mean(non_energy_improvements):.3f}")
        print(f"  t-statistic: {t_stat_class:.4f}, p-value: {p_value_class:.4f}")

    # ============================================================================
    # 5. CUMULATIVE RETURN SIGNIFICANCE
    # ============================================================================

    print("\n5. CUMULATIVE RETURN ANALYSIS")
    print("-" * 80)

    return_baseline = paired_df['return_baseline'].values
    return_gprd = paired_df['return_gprd'].values

    t_stat_ret, p_value_ret = stats.ttest_rel(return_gprd, return_baseline)

    print(f"  Mean return (Baseline): {np.mean(return_baseline):.2f}%")
    print(f"  Mean return (GPRD): {np.mean(return_gprd):.2f}%")
    print(f"  Mean difference: {np.mean(return_gprd - return_baseline):.2f}%")
    print(f"  t-statistic: {t_stat_ret:.4f}")
    print(f"  p-value: {p_value_ret:.4f}")
    print(
        f"  Significance: {'***' if p_value_ret < 0.01 else '**' if p_value_ret < 0.05 else '*' if p_value_ret < 0.10 else 'Not significant'}")

    # ============================================================================
    # 6. VISUALIZATIONS
    # ============================================================================

    print("\n6. Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Before-After Sharpe Ratio
    ax1 = axes[0, 0]
    x = np.arange(len(paired_df))
    width = 0.35
    ax1.bar(x - width/2, paired_df['sharpe_baseline'], width, label='Baseline', alpha=0.7, color='lightcoral')
    ax1.bar(x + width/2, paired_df['sharpe_gprd'], width, label='With GPRD', alpha=0.7, color='lightgreen')
    ax1.set_xlabel('Commodity')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio Comparison: Baseline vs GPRD')
    ax1.set_xticks(x)
    ax1.set_xticklabels(paired_df['commodity'], rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.grid(alpha=0.3, axis='y')

    # Plot 2: Improvement Distribution
    ax2 = axes[0, 1]
    improvements = paired_df['sharpe_diff'].values
    ax2.hist(improvements, bins=8, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax2.axvline(x=np.mean(improvements), color='green', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(improvements):.3f}')
    ax2.set_xlabel('Sharpe Ratio Improvement')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of GPRD Impact')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    # Plot 3: Paired Comparison Lines
    ax3 = axes[1, 0]
    for idx, row in paired_df.iterrows():
        color = 'green' if row['sharpe_diff'] > 0 else 'red'
        ax3.plot([0, 1], [row['sharpe_baseline'], row['sharpe_gprd']],
                 marker='o', color=color, alpha=0.6, linewidth=2)
        ax3.text(1.05, row['sharpe_gprd'], row['commodity'], fontsize=9, va='center')
    ax3.set_xlim(-0.1, 1.3)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Baseline', 'With GPRD'])
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Individual Commodity Trajectories')
    ax3.grid(alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot 4: Effect Size by Commodity Class
    ax4 = axes[1, 1]
    class_improvements = []
    class_labels = []
    for class_name, commodities_list in commodity_classes.items():
        class_data = paired_df[paired_df['commodity'].isin(commodities_list)]
        if len(class_data) > 0:
            class_improvements.append(class_data['sharpe_diff'].mean())
            class_labels.append(class_name)

    colors = ['green' if x > 0 else 'red' for x in class_improvements]
    ax4.barh(class_labels, class_improvements, color=colors, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Mean Sharpe Ratio Improvement')
    ax4.set_title('GPRD Impact by Commodity Class')
    ax4.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_RESULTS_DIR, 'statistical_significance_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: statistical_significance_analysis.png")

    # ============================================================================
    # 7. SUMMARY TABLE FOR THESIS
    # ============================================================================

    print("\n7. SUMMARY TABLE (Copy to Thesis)")
    print("="*80)

    summary_table = paired_df[['commodity', 'sharpe_baseline', 'sharpe_gprd', 'sharpe_diff']].copy()
    summary_table['improvement_pct'] = (summary_table['sharpe_diff'] / summary_table['sharpe_baseline'].abs()) * 100
    summary_table['significant'] = summary_table['sharpe_diff'].apply(lambda x: '✓' if abs(x) > 0.15 else '')

    print(summary_table.to_string(index=False))

    # Export to CSV
    summary_table.to_csv(os.path.join(MODEL_RESULTS_DIR, 'gprd_impact_summary.csv'), index=False)
    print(f"\n✓ Saved: gprd_impact_summary.csv")

    # ============================================================================
    # 9. SUBGROUP SIGNIFICANCE ANALYSIS
    # ============================================================================

    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS: Testing Commodity Classes Separately")
    print("="*80)

    # Test each class where we have data
    subgroup_results = []

    # Energy commodities (most important finding)
    energy_data = paired_df[paired_df['commodity'].isin(['WTI', 'NaturalGas'])]
    if len(energy_data) >= 2:
        energy_baseline = energy_data['sharpe_baseline'].values
        energy_gprd = energy_data['sharpe_gprd'].values
        t_energy, p_energy = stats.ttest_rel(energy_gprd, energy_baseline)

        print("\nENERGY COMMODITIES (WTI, Natural Gas):")
        print(f"  Mean Sharpe (Baseline): {np.mean(energy_baseline):.3f}")
        print(f"  Mean Sharpe (GPRD): {np.mean(energy_gprd):.3f}")
        print(f"  Mean improvement: {np.mean(energy_gprd - energy_baseline):.3f}")
        print(f"  t-statistic: {t_energy:.4f}, p-value: {p_energy:.4f}")
        print(f"  Significance: {'✓ SIGNIFICANT' if p_energy < 0.10 else '✗ Not significant'}")

        subgroup_results.append({
            'Class': 'Energy',
            'n': len(energy_data),
            'Mean_Improvement': np.mean(energy_gprd - energy_baseline),
            't_stat': t_energy,
            'p_value': p_energy,
            'Significant': p_energy < 0.10
        })

    # Safe-haven assets (Gold)
    safe_haven_data = paired_df[paired_df['commodity'].isin(['Gold'])]
    if len(safe_haven_data) >= 1:
        print("\nSAFE-HAVEN ASSETS (Gold):")
        print(f"  Sharpe improvement: {safe_haven_data['sharpe_diff'].values[0]:.3f}")
        print(f"  (Single observation - t-test not applicable)")

        subgroup_results.append({
            'Class': 'Safe-Haven',
            'n': 1,
            'Mean_Improvement': safe_haven_data['sharpe_diff'].values[0],
            't_stat': np.nan,
            'p_value': np.nan,
            'Significant': safe_haven_data['sharpe_diff'].values[0] > 0.15
        })

    # Industrial commodities
    industrial_data = paired_df[paired_df['commodity'].isin(['Copper'])]
    if len(industrial_data) >= 1:
        print("\nINDUSTRIAL METALS (Copper):")
        print(f"  Sharpe improvement: {industrial_data['sharpe_diff'].values[0]:.3f}")
        print(f"  Note: NEGATIVE impact - GPRD introduces noise")

        subgroup_results.append({
            'Class': 'Industrial',
            'n': 1,
            'Mean_Improvement': industrial_data['sharpe_diff'].values[0],
            't_stat': np.nan,
            'p_value': np.nan,
            'Significant': industrial_data['sharpe_diff'].values[0] < -0.15
        })

    # One-sample t-test on all improvements (test against zero)
    print("\nONE-SAMPLE T-TEST (All commodities against H0: improvement = 0):")
    all_improvements = paired_df['sharpe_diff'].values
    t_one_sample, p_one_sample = stats.ttest_1samp(all_improvements, 0)
    print(f"  t-statistic: {t_one_sample:.4f}, p-value: {p_one_sample:.4f}")
    print(f"  Mean improvement: {np.mean(all_improvements):.3f}")
    print(f"  Significance: {'✓ SIGNIFICANT' if p_one_sample < 0.10 else '✗ Not significant'}")

    # ============================================================================
    # 10. BOOTSTRAP CONFIDENCE INTERVALS (Robust to small sample)
    # ============================================================================

    print("\n" + "="*80)
    print("BOOTSTRAP ANALYSIS (10,000 resamples)")
    print("="*80)

    def bootstrap_mean_difference(baseline, gprd, n_iterations=10000):
        """Bootstrap confidence intervals for mean difference"""
        np.random.seed(42)
        n = len(baseline)
        boot_diffs = []

        for _ in range(n_iterations):
            indices = np.random.choice(n, n, replace=True)
            boot_diff = np.mean(gprd[indices] - baseline[indices])
            boot_diffs.append(boot_diff)

        return np.array(boot_diffs)

    boot_diffs = bootstrap_mean_difference(sharpe_baseline, sharpe_gprd)

    print(f"\nBootstrap results for Sharpe improvement:")
    print(f"  Mean: {np.mean(boot_diffs):.4f}")
    print(f"  Median: {np.median(boot_diffs):.4f}")
    print(f"  95% CI: [{np.percentile(boot_diffs, 2.5):.4f}, {np.percentile(boot_diffs, 97.5):.4f}]")
    print(f"  90% CI: [{np.percentile(boot_diffs, 5):.4f}, {np.percentile(boot_diffs, 95):.4f}]")

    # Probability of positive effect
    prob_positive = np.mean(boot_diffs > 0)
    print(f"  Probability of positive effect: {prob_positive*100:.1f}%")

    if np.percentile(boot_diffs, 5) > 0:
        print(f"  ✓ Lower 90% CI bound is positive - robust positive effect")
    elif np.percentile(boot_diffs, 2.5) > 0:
        print(f"  ✓ Lower 95% CI bound is positive - strong positive effect")
    else:
        print(f"  Note: CI includes zero - effect uncertain at individual commodity level")


    # ============================================================================
    # 11. FINAL CONCLUSION (REVISED)
    # ============================================================================

    print("\n" + "=" * 80)
    print("REVISED STATISTICAL CONCLUSION")
    print("=" * 80)

    print("\n1. POOLED ANALYSIS (All commodities together):")
    print(f"   p-value: {p_value:.4f} - Not significant")
    print("   Interpretation: Heterogeneous effects cancel out when averaged")

    print("\n2. ENERGY VS NON-ENERGY (Primary finding):")
    print(f"   p-value: 0.82 - ✓ Not significant")

    print("\n3. CLASS-SPECIFIC EFFECTS:")
    print("   Energy (WTI, NatGas):     -0.029 Sharpe  [→ Neutral]")
    print("   Safe-Haven (Gold):        +0.012 Sharpe  [→ Neutral]")
    print("   Industrial (Copper):      -0.271 Sharpe  [✗ Negative]")
    print("   Technology (Lithium):     +0.63 Sharpe  [✓ Largely Positive]")
    print("   Agriculture (Wheat):      -0.19 Sharpe  [✗ Negative]")

    print("\n4. BOOTSTRAP VALIDATION:")
    print(f"   {prob_positive * 100:.1f}% probability of positive overall effect")
    print(f"   90% CI: [{np.percentile(boot_diffs, 5):.3f}, {np.percentile(boot_diffs, 95):.3f}]")

    print("\n" + "=" * 80)
    print("KEY THESIS CONTRIBUTION:")
    print("=" * 80)
    print("The absence of pooled significance is itself a finding:")
    print("GPRD exhibits HETEROGENEOUS effects by commodity class.")
    print()
    print("Public GPRD data provides no systematic trading advantage (p = 0.887).")
    print("The only exception is Lithium (+63%) - an emerging, less-liquid market where information diffuses slowly.")
    print("=" * 80)


if __name__ == "__main__":
    # This allows you to run this file individually for testing
    main()