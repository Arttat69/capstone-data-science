---
title: "Geopolitics and Commodities: From Predictive Forecasting to Statistical Association"
author: "Taton Arthur (Arthur.Taton@unil.ch)"
date: "December 2025"
institution: "HEC Lausanne"
course: "Data Science and Advanced Programming - Final Project"
---

## Abstract

Commodity markets often react sharply to wars, sanctions, and diplomatic crises. This study tests whether geopolitical risk measures improve commodity forecasting using a dual approach. Using six commodities—WTI Oil, Natural Gas, Gold, Wheat, Copper, and Lithium—from January 2003 to November 2024, we first evaluate Long Short-Term Memory (LSTM) models augmented with the Caldara–Iacoviello Geopolitical Risk Index (GPRD). Out-of-sample R² is predominantly negative across all commodities and feature sets, implying limited practical predictive value for trading once realistic frictions are considered. Given this forecasting failure, we pivot to GARCH-in-mean models to test statistical association rather than trading utility. **Geopolitical regressors are jointly significant across all commodities (p < 0.001 in all 18 comparisons), while individual coefficients are universally insignificant**—consistent with multicollinearity between GPRD and news-derived variables and economically small effects. With 2,800–5,300 daily observations per commodity, tests detect tiny associations that remain too small to exploit. Heterogeneity is meaningful: Lithium shows the strongest sensitivity (1.38% AIC improvement), followed by Copper (1.07%) and Wheat (0.95%), while energy commodities show weaker associations (0.60–0.64%). Overall, geopolitical risk is statistically detectable in returns but typically too weak and noisy for reliable prediction or profitable strategies.

**Keywords:** commodity forecasting, geopolitical risk, LSTM neural networks, GARCH models, statistical versus practical significance

---

## Introduction

Commodity prices respond directly to physical supply–demand imbalances, making them unusually sensitive to geopolitical disruptions. The 2022 invasion of Ukraine illustrates the mechanism: European natural gas prices surged around 300% as supply was reshuffled, and wheat prices rose roughly 50% as exports were disrupted and production became constrained in conflict zones.

Such episodes recur (oil embargoes, wars, sanctions, and regional instability), highlighting a core forecasting limitation: models driven only by price history struggle with exogenous shocks that shift supply curves and trigger rapid repricing. Standard technical approaches treat markets as largely self-contained and therefore cannot anticipate external political events.

A practical opportunity emerged with better measurement. Caldara and Iacoviello (2022) construct a daily Geopolitical Risk Index (GPRD) from newspaper text across major international outlets, providing high-frequency proxies for global tension. If markets do not fully and immediately price geopolitical information, incorporating GPRD into forecasting models could, in principle, improve performance by converting "surprise" volatility into partially forecastable patterns.

The initial hypothesis was straightforward: deep learning models enriched with geopolitical risk features might forecast commodity dynamics well enough to generate tradable signals after costs. LSTM architectures were trained across six commodities (energy, metals, agriculture), using price-based features and GPRD-derived indicators. Models forecast 5-day rolling volatility, which was then mapped into trading positions.

Empirically, the forecasting approach fails broadly: LSTM variants—baseline and geopolitically enhanced—produce negative out-of-sample R² in most cases, meaning they underperform a naive mean benchmark. The consistency of failure across assets and configurations suggests structural difficulty rather than a tunable implementation issue.

This created a key interpretation problem. "No profitable forecasting" does not imply "no relationship." It answers a trading question (economic exploitability), not the econometric question (statistical association). Therefore, the project pivots to GARCH-in-mean models with geopolitical regressors, separating forecasting performance from hypothesis testing. This distinction matters: large samples can produce low p-values even when effects are economically negligible.

---

## Research Question & Literature

### Research questions and contributions

This report addresses three linked questions:

**RQ1 (Predictive Utility):** Do geopolitical indices improve out-of-sample forecasting and trading performance versus price-only baselines?

**RQ2 (Statistical Association):** Is there any detectable relationship between geopolitical risk and commodity returns, even if too small for trading?

**RQ3 (Heterogeneity):** Do effects differ systematically across commodity sectors (energy, agriculture, metals)?

Contributions:
- Full reporting of null forecasting results (mitigating publication bias toward positive ML backtests).
- Clear separation of statistical significance (p-values) from practical significance (economic value).
- Commodity-level heterogeneity rather than pooled inference that hides sector differences.
- An interpretation consistent with efficient markets: fast incorporation leaves only small residual signals.

### Related work and context

**Machine learning for commodity forecasting.**  
Commodity forecasting moved from classical ARIMA/GARCH frameworks toward machine learning, including LSTMs, which can represent nonlinear sequential relationships and long-memory patterns. LSTMs address vanishing gradients via gating (input/forget/output), enabling selective information retention over longer horizons than basic RNNs.

However, reported performance often deteriorates under rigorous out-of-sample testing. Many studies emphasize in-sample fit but use weak validation designs or omit realistic frictions. Related evidence (e.g., Fischer and Krauss, 2018) shows that strong backtests frequently fail once costs, slippage, and non-stationarity are considered. Two mechanisms dominate: overfitting to idiosyncratic historical noise and regime shifts that invalidate "learned" relationships.

**Geopolitical risk measurement and market impact.**  
Caldara and Iacoviello's (2022) GPRD is built from automated counts of threat-related terms in leading newspapers, producing a daily series that spikes during major crises (wars, terrorism, nuclear events). This provides a standardized proxy for geopolitical intensity suitable for empirical testing.

Empirical findings on GPRD–commodity links are mixed. A plausible explanation is that effects are real but small and heterogeneous: some commodities respond through supply disruptions (energy/agriculture), while others are driven mainly by global demand cycles (industrial metals). In large datasets, small effects can be statistically detected without being large enough to trade.

**Statistical versus practical significance.**  
Methodological work emphasizes that p-values quantify detectability under a null, not economic magnitude. With large n, even very small effects can yield extreme significance. In financial contexts, this creates a common mistake: interpreting "statistically significant" as "tradable."

A useful framing is signal-to-noise. If daily volatility is 50–200 bps, then predictors that shift expected returns by only a few basis points can be real yet unexploitable after costs. This report uses that lens to reconcile strong GARCH inference with weak forecasting utility.

---

## Methodology

### Data sources and preprocessing

The sample spans January 2003 to November 2024 (~21 years; ~5,300 trading days), covering multiple geopolitical regimes (Iraq War, Arab Spring, Crimea, trade tensions, COVID, Ukraine). Daily prices are retrieved via yfinance for six assets: WTI (CL=F), Natural Gas (UNG), Gold (GC=F), Wheat (ZW=F), Copper (HG=F), and Lithium (LIT, from July 2010). GPRD is obtained from FRED and aligned to trading days via forward filling.

Returns are log differences: Return_t = log(Close_t / Close_{t-1}). Volatility is 5-day rolling standard deviation Vol_5 = std(Return_{t-4:t}). Missing values are limited (<1% except Lithium due to late inception) and handled with short forward fills and interpolation where needed.

Enhanced LSTM features total 21 variables: lagged returns/volatility, EMAs, raw and lagged GPRD measures, GPRD regimes (e.g., above the 75th percentile), and news-derived counts/flags. All inputs are lagged to prevent lookahead bias. RobustScaler normalization reduces sensitivity to outliers and stabilizes NN training. Correlations show substantial overlap between GPRD and news features (r > 0.65), motivating caution in interpreting individual regression coefficients later.

### Data integration

Price data for Gold, WTI, Wheat, Natural Gas, Copper, and Lithium is pulled via yfinance, flattened, de-duplicated on Date, and enriched with Return, MA_5, Vol_5, and return lags; processed CSVs are saved per commodity.

The daily Geopolitical Risk Index (GPRD) is loaded, cleaned, resampled to calendar days with forward-fill, and merged on Date; enriched datasets additionally align GPRD and aggregated news features using merge_asof to avoid lookahead.

ABC News headlines are parsed to daily dates, then aggregated to a per-day sum of geo_keyword_hits and mean VADER sentiment, which are merged into each commodity's panel and trimmed to the study window before saving.

### Crisis lexicon

The crisis lexicon spans conflict (war, invasion, terrorism), policy/economic stress (sanctions, embargo, tariffs, recession), supply/logistics (blockade, pipeline), and geopolitical entities/events (OPEC, NATO, Ukraine, Gaza, Crimea, Arab Spring), among others.

geo_keyword_hits counts daily occurrences of these tokens in headlines, serving as a compact news-intensity proxy aligned to each trading day, while VADER's compound score captures average headline sentiment per day.

### Stage 1: LSTM neural network forecasting

We run A/B tests across multiple feature sets:

- **Baseline LSTM:** 15-day sequences, 6 features (lags of returns and volatility), two LSTM layers (32 then 16 units), dropout 0.2; 6,657 parameters.
- **GPRD LSTM:** Adds GPRD-derived features (12 total features)
- **GEO LSTM:** Adds geo_keyword_hits features (12 total features)
- **Combined LSTM:** All geopolitical features (15 total features)
- **Granger-Optimal LSTM:** Uses only statistically significant lags from Granger causality tests

For enhanced models, network size is reduced (24 then 8 units) to control overfitting; 5,969 parameters.

Training uses Huber loss, Adam (lr=0.001), early stopping (patience 15), and learning-rate reduction. Data is split by time: train 2003–2020 (80%), test 2020–2024 (20%).

Predicted volatility is translated into positions: above 75th percentile → short -0.5; mid-range → long +0.5; below 25th percentile → long +1.0. Transaction costs of 0.1% per trade approximate fees and slippage.

Evaluation metrics: out-of-sample R², annualized Sharpe ratio, maximum drawdown, and significance testing via paired t-tests and bootstrap resampling.

### Stage 2: GARCH econometric modeling

We estimate GARCH(1,1) with mean equations that either exclude or include geopolitical regressors. Baseline mean: Return_t = μ + ε_t, with σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}. Enhanced mean adds geo_keyword_hits and/or GPRD: Return_t = μ + β₁·geo_keyword_hits_t + β₂·GPRD_t + ε_t.

Estimation uses maximum likelihood (arch library) with robust errors. Nested specifications are compared with likelihood ratio tests, and AIC/BIC quantify fit improvements penalized for complexity. We focus on (i) joint significance and (ii) whether individual coefficients remain significant once correlated regressors enter together.

---

## Implementation

Key technical decisions include: all features lagged (t-1 to t-3) to prevent lookahead bias; RobustScaler normalization to handle heavy tails and outliers; intentionally smaller enhanced LSTM network (24→8 vs 32→16 units) to limit overfitting in high-dimensional feature space; and 0.1% transaction costs per trade to ensure realistic backtest conditions.

### Modeling approaches

Classical regression: a baseline Linear Regression uses Return_lag1, while an enhanced RandomForestRegressor uses Return_lag1 plus GPRD, geo_keyword_hits, sentiment, and 1–5 day lags of GPRD and geo_keyword_hits (scaled with StandardScaler).

Classification: a Logistic Regression predicts the sign of next-day returns using the enhanced feature set after splitting on time (train < 2018-01-01, test ≥ 2018-01-01), with Return_binary constructed as 1 if Return > 0.

Regime clustering: KMeans(n=2) on Vol_5, GPRD, and geo_keyword_hits segments calm/stressed regimes and supports subsequent regime-aware diagnostics and visualizations.

A typical failure mode emerged: baseline LSTMs collapse to near-constant volatility predictions (minimal variance), which reduces training loss but provides no trading usefulness. Enhanced models produce time-varying forecasts but fail to generalize out-of-sample, evidenced by negative R² across most commodities. This motivated the econometric pivot to GARCH models to test for statistical association independent of forecasting utility.

---

## Codebase & Reproducibility

To reproduce results: (1) install dependencies from `requirements.txt`, (2) configure a Kaggle API token (`kaggle.json`) if news dataset download is required, and (3) run the `main` Python file, which executes the full pipeline (data preprocessing, LSTM training, GARCH estimation, evaluation, and result exports). Repository: https://github.com/Arttat69/capstone-data-science.git

---

## Results

### LSTM forecasting performance

Out-of-sample R² indicates broad forecasting failure across both baseline and enhanced LSTM models:

**Baseline LSTM (price-only features):**
- Gold: -0.061
- WTI: -0.016
- Wheat: -0.023
- Natural Gas: -0.465
- Copper: -0.087
- Lithium: -0.638

**Enhanced LSTM - COMBINED (GPRD + geo_keyword_hits + sentiment):**
- Gold: -0.067
- WTI: -0.015
- Wheat: -0.011
- Natural Gas: -0.455
- Copper: -0.110
- Lithium: -0.628

Natural Gas and Lithium show the worst forecasting performance, with R² below -0.45, meaning predictions are substantially worse than using the historical mean. The enhanced models show marginal improvements for some commodities (Wheat improves from -0.023 to -0.011, Natural Gas from -0.465 to -0.455) but remain deeply negative overall, indicating no reliable predictive signal.

**Feature Set Comparison:**
Testing five feature configurations (Baseline, GPRD, GEO, Combined, Granger-Optimal) shows no consistent winner:

- Wheat shows modest improvement with geopolitical features (all enhanced variants achieve -0.011 vs baseline -0.023)
- Copper's best R² is with Granger-optimal features (-0.063), but this still represents poor predictive power
- Natural Gas remains deeply negative across all specifications (-0.441 to -0.465)
- No commodity achieves positive out-of-sample R² with any feature set

Diagnostics show a typical failure mode: baseline models often collapse to near-constant volatility predictions (minimal variance), which reduces training loss but provides no trading usefulness. Enhanced models avoid this collapse and produce time-varying forecasts, but negative R² implies that this variability does not generalize out-of-sample.

### Additional model results

Classification accuracy for each commodity is very close to 0.5 (range: 0.503–0.524), indicating near coin-flip directionality under the tested features and split.

RMSE for Linear Regression and Random Forest are both very close for each commodity (around 0.02), with WTI as the outlier at roughly 0.1.

These results are consistent with weak short-horizon predictability and suggest limited incremental value from daily GPRD/news features for price prediction once realistic splits and scaling are enforced.

### Trading backtest and statistical tests

Trading simulations reveal no systematic improvement from geopolitical features:

**Sharpe Ratio Performance:**
- Mean baseline Sharpe: 0.123
- Mean enhanced Sharpe (COMBINED): 0.145
- Mean improvement: +0.022

**Statistical testing:**
- Paired t-test: t = 0.1495, p = 0.8870 (not significant)
- Wilcoxon test: statistic = 7.0, p = 0.5625 (not significant)
- Cohen's d: 0.0668 (negligible effect)
- Bootstrap 90% CI: [-0.148, 0.235]
- Probability of positive effect: 54.0%

The pooled analysis finds no significant improvement. However, this masks strong heterogeneity:

**Commodity-specific Sharpe changes (Baseline vs Best Enhanced):**
- Gold: +0.012 (0.681 → 0.681, essentially unchanged)
- WTI: +0.054 (COMBINED beats baseline: -0.414 → -0.360)
- Wheat: +0.098 (0.211 → 0.310, moderate improvement)
- Natural Gas: +0.186 (Granger-optimal: -0.092 → 0.095)
- Copper: +0.059 (GEO features: 0.404 → 0.463)
- Lithium: essentially unchanged (-0.052 baseline, -0.052 for most enhanced variants)

**Important caveat:** Even these "improvements" must be interpreted cautiously—most commodities still have negative or barely positive Sharpe ratios, and improvements are not statistically significant at the pooled level.

**Subgroup analysis:**
- Energy commodities (WTI, Natural Gas): mean baseline Sharpe = 0.134, mean GPRD Sharpe = 0.105, difference = -0.029, t = -1.85, p = 0.316 (not significant, negative direction)
- The initial hypothesis of energy commodity improvement is not supported

### GARCH statistical associations

**Understanding GARCH Results: Joint vs Individual Significance**

The GARCH analysis reveals a critical statistical paradox that explains why geopolitical risk is detectable but not tradable:

#### What the Tests Show

**1. Likelihood Ratio Tests (Joint Significance) - ALL HIGHLY SIGNIFICANT:**

Every single commodity shows that geopolitical variables jointly improve model fit at extreme significance levels (p < 0.001):

| Commodity   | LR Statistic | p-value    | Interpretation |
|-------------|--------------|------------|----------------|
| Wheat       | 181.01       | < 10⁻⁴⁰    | Extreme significance |
| Copper      | 180.57       | < 10⁻⁴⁰    | Extreme significance |
| Lithium     | 129.10       | < 10⁻²⁸    | Extreme significance |
| WTI         | 120.36       | < 10⁻²⁶    | Extreme significance |
| Natural Gas | 99.58        | < 10⁻²²    | Extreme significance |
| Gold        | 92.44        | < 10⁻²⁰    | Extreme significance |

**What this means:** When we test whether adding both geo_keyword_hits AND GPRD together improves the model, the answer is an emphatic YES for all six commodities. The probability that these variables have zero effect is essentially impossible (p-values so small they're effectively zero).

**2. Individual Coefficient Tests - NONE SIGNIFICANT:**

Yet when we look at whether each variable matters *on its own* (controlling for the other), we find:

**Gold (Combined Model):**
- geo_keyword_hits: coefficient = 0.001, p = 0.944 (not significant)
- GPRD: coefficient = -0.016, p = 0.186 (not significant)

**WTI (Combined Model):**
- geo_keyword_hits: coefficient = 0.005, p = 0.874 (not significant)
- GPRD: coefficient = 0.030, p = 0.354 (not significant)

**Wheat (Combined Model):**
- geo_keyword_hits: coefficient = -0.014, p = 0.602 (not significant)
- GPRD: coefficient = 0.003, p = 0.907 (not significant)

**Natural Gas (Combined Model):**
- geo_keyword_hits: coefficient = 0.013, p = 0.773 (not significant)
- GPRD: coefficient = -0.026, p = 0.515 (not significant)

**Copper (Combined Model):**
- geo_keyword_hits: coefficient = 0.025, p = 0.227 (not significant)
- GPRD: coefficient = -0.007, p = 0.689 (not significant)

**Lithium (Combined Model):**
- geo_keyword_hits: coefficient = -0.046, p = 0.132 (not significant)
- GPRD: coefficient = 0.002, p = 0.948 (not significant)

**Result:** Not a single individual coefficient is significant (p > 0.10) across all six commodities.

#### Why This Paradox Occurs

This is not a contradiction—it reflects three fundamental statistical realities:

**1. Multicollinearity:**
- GPRD and geo_keyword_hits are highly correlated (r = 0.65–0.78)
- They measure overlapping concepts: both capture geopolitical tension
- When both enter the model together, it's hard to determine which one "deserves credit" for the explanatory power
- This inflates standard errors on individual coefficients, making them appear insignificant
- But jointly they still provide signal

**2. Large Sample Power:**
- With 2,800–5,800 observations per commodity, likelihood ratio tests have enormous statistical power
- They can detect tiny improvements in log-likelihood that cumulate into large test statistics
- The joint test asks: "Do these variables improve fit at all?"
- With thousands of observations, even a 0.1% improvement in fit produces extreme significance

**3. Economic Magnitude vs Statistical Significance:**
- The coefficient sizes are economically tiny (0.001–0.046)
- A one-standard-deviation shock in GPRD shifts expected returns by only 0.5–5 basis points
- Daily volatility is 50–200 basis points—an order of magnitude larger
- After 10 bps transaction costs, these effects are untradable
- But they're still statistically detectable with large n

#### AIC Model Comparison

AIC (Akaike Information Criterion) penalizes model complexity, so improvements indicate genuine explanatory value:

**AIC Improvements (Baseline vs Best Geopolitical Model):**
- Lithium: 1.38% reduction (127 AIC points)
- Copper: 1.07% reduction (178 AIC points)
- Wheat: 0.95% reduction (179 AIC points)
- Gold: 0.71% reduction (90 AIC points)
- Natural Gas: 0.64% reduction (98 AIC points)
- WTI: 0.60% reduction (118 AIC points)

**Interpretation:** All commodities show consistent AIC improvements when geopolitical variables are added, confirming they contain real (if small) information. The ranking differs from the initial hypothesis—industrial metals and agriculture show stronger associations than energy.

#### Visual Evidence from Appendix G

Volatility time-series plots (Appendix G) show that GPRD spikes often coincide with commodity volatility increases during major crises (2008 financial crisis, COVID-19, Ukraine invasion). However:
- Many high-GPRD periods do not trigger volatility spikes
- Some volatility spikes occur without elevated GPRD
- The relationship is "necessary but not sufficient"—geopolitical stress is one factor among many

This visual imperfection helps explain why models detect association yet fail at reliable forecasting.

### Reconciling the statistical-practical paradox

**The Bottom Line:** 

Geopolitical risk is **statistically real** but **economically weak**:

✅ **Statistical Reality (p < 0.001):**
- Joint tests prove geopolitical variables improve model fit
- This is consistent across all six commodities
- Large samples make even tiny effects detectable

❌ **Economic Reality (not tradable):**
- Effect sizes are only 0.5–5 basis points
- Daily noise is 50–200 basis points (10-40x larger)
- Transaction costs (~10 bps) exceed signal strength
- Multicollinearity prevents isolating which specific variable matters

**This is not a failure of the analysis—it's an important finding:**
Markets are reasonably efficient. If GPRD contained large, persistent, exploitable predictive power, sophisticated traders would have arbitraged it away. What remains is a small residual association: real enough to detect with good data and proper tests, but too small to trade profitably at daily frequency.

The paradox actually validates market efficiency: geopolitical news gets incorporated quickly (likely within minutes/hours via real-time feeds and alternative data), leaving only tiny scraps of predictable information by daily close.

### Commodity heterogeneity and economic mechanisms

Heterogeneity is economically interpretable:

- **Industrial Metals (Lithium, Copper):** Highest AIC sensitivity (1.38% and 1.07%), suggesting that supply chain disruptions and technology/infrastructure investment cycles create moderate geopolitical linkages. Lithium's emergence as a strategic resource for battery technology may explain its leading sensitivity.

- **Agriculture (Wheat):** Strong sensitivity (0.95%) reflects concentrated production/export routes and vulnerability to conflict-related logistics shocks (e.g., Black Sea disruptions during Ukraine crisis).

- **Precious Metals (Gold):** Moderate sensitivity (0.71%), reflecting selective safe-haven behavior during systemic crises rather than localized conflicts.

- **Energy (WTI, Natural Gas):** Lower sensitivity (0.60–0.64%) than initially hypothesized. Energy markets may already incorporate geopolitical risk premiums through other channels (futures curves, inventory signals, real-time news), leaving less residual information in daily GPRD. Alternatively, diversified supply sources and strategic reserves may buffer short-term geopolitical shocks.

---

## Conclusion

### Summary of key findings

This project delivers four main findings. 

**First,** LSTM forecasting—baseline and geopolitically enhanced—generally fails out-of-sample, with negative R² across all commodities and feature sets. Even the best-performing specifications achieve only marginal improvements (Wheat: -0.023 to -0.011), implying little practical predictive value for daily trading once frictions are considered.

**Second,** GARCH-in-mean models detect consistent geopolitical association: all 18 likelihood ratio tests reject zero-effect nulls at p < 0.001, and AIC improves by 0.60%–1.38% across commodities. This represents clear statistical evidence that geopolitical variables contain information about commodity returns.

**Third,** the paradox of joint significance without individual significance reflects (i) multicollinearity between GPRD and news features (r > 0.65), (ii) economically tiny coefficients that shift expected returns by only a few basis points, and (iii) large-sample power that detects minuscule effects. This is not a methodological failure but an economically meaningful finding: geopolitical risk is statistically real but too small relative to daily volatility and transaction costs to exploit profitably.

**Fourth,** sector heterogeneity differs from initial expectations: industrial metals (Lithium 1.38%, Copper 1.07%) and agriculture (Wheat 0.95%) show the strongest sensitivity, while energy commodities show weaker associations (Natural Gas 0.64%, WTI 0.60%) than hypothesized. Gold (0.71%) responds selectively to systemic stress rather than all geopolitical events.

### Practical implications

For traders, GPRD is not a reliable standalone signal at daily frequency. The statistically significant associations detected in GARCH models translate to effect sizes (0.5–5 bps) that are an order of magnitude smaller than typical daily volatility (50–200 bps) and comparable to or smaller than realistic transaction costs. Limited evidence suggests some commodities (Wheat, Natural Gas in specific feature sets) may benefit marginally from geopolitical features in broader ensembles, but gains are not robust or statistically significant at the portfolio level.

For risk management, GPRD remains useful as a regime indicator: major crises coincide with volatility surges even when exact timing is hard to predict. Stress testing and VaR frameworks should account for fat tails during elevated geopolitical periods, and commodity-specific exposure matters—wheat and industrial metals show stronger sensitivity to supply disruptions than energy markets in daily frequency data.

For researchers, results reinforce methodological best practices: 
- Prioritize out-of-sample validation over in-sample fit
- Report effect sizes alongside p-values (a p < 0.001 finding with a 2 bps effect is not tradable)
- Analyze commodity heterogeneity instead of pooling all assets
- Distinguish between statistical detectability and economic exploitability
- Use likelihood ratio tests for joint significance when predictors are correlated
- Consider that "no profitable forecasting" ≠ "no relationship"

The finding that joint tests are highly significant while individual coefficients are not should be reported as a feature, not a bug—it demonstrates that the analysis properly accounts for multicollinearity and large-sample statistical power.

### Limitations

The commodity set is small (six assets), Lithium has a shorter history (from 2010), and the analysis uses daily frequency (information may be priced faster intraday). GPRD is a global index and may dilute region-specific commodity shocks (e.g., Middle East tensions for oil, US-China relations for technology metals). Transaction costs are approximated at 0.1% per trade, which may understate real frictions for retail traders or overstate them for institutions. The news dataset (ABC headlines) captures only English-language U.S. media and may miss important international coverage. Finally, the study period includes extraordinary events (2008 crisis, COVID-19) that may not be representative of normal market conditions.

### Future research directions

Future work should test (i) alternative architectures (e.g., attention/Transformers) that can dynamically weight geopolitical lags, (ii) commodity-specific geopolitical indices (Middle East tensions for oil, US-China trade for tech materials, Black Sea stability for wheat) rather than global GPRD, and (iii) higher-frequency data (hourly/intraday) where short-lived mispricings may exist before daily close. 

Event-study designs with carefully defined shocks (e.g., using instrumental variables or natural experiments around unexpected geopolitical events) could isolate clearer causal effects than broad daily indices. Panel methods that pool across commodities while allowing heterogeneous coefficients might improve power while respecting sectoral differences.

Finally, combining geopolitical signals with other information sets (inventory data, shipping flows, satellite imagery of production facilities) in a multi-modal framework might capture the complete channel through which geopolitics affects prices—our findings suggest GPRD alone captures only a small fraction of the total effect.

### Final assessment

Geopolitical indices do not provide reliable daily forecasting power for tradable commodity strategies in this setting. However, they do exhibit statistically detectable associations with returns and volatility dynamics, with economically interpretable heterogeneity across sectors. 

The practical lesson is not that geopolitics "doesn't matter," but that:
1. Most of its daily measurable signal is either rapidly priced or too small relative to noise and costs
2. What remains is statistically significant (p < 0.001 across all commodities) but economically tiny (0.5–5 bps vs 50–200 bps daily volatility)
3. Different commodities respond through different mechanisms (supply shocks for wheat, safe-haven flows for gold, demand cycles for industrial metals)
4. Market efficiency implies predictable patterns are either arbitraged away or reflect compensation for risk rather than exploitable mispricings

The divergence between statistical and practical significance, rather than being a methodological problem, validates the hypothesis that markets incorporate public geopolitical information efficiently—leaving only small residual associations that are detectable with proper econometric tools but insufficient for profitable trading strategies.

---

## Appendix: AI Tools Used

For details on AI tools used during this project (including specific tools, tasks, and usage contexts), please refer to `AI_USAGE.md` in the project repository.

---

## References

Caldara, D., & Iacoviello, M. (2022). Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194-1225.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.

Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

McShane, B. B., et al. (2019). Abandon Statistical Significance. *The American Statistician*, 73(sup1), 235-245.

---

## Appendices (Project Material)

### Appendix A: Commodity Ticker Symbols

- Gold: GC=F (COMEX Gold Futures)
- WTI Oil: CL=F (NYMEX Crude Oil Futures)
- Wheat: ZW=F (CBOT Wheat Futures)
- Natural Gas: UNG (United States Natural Gas ETF)
- Copper: HG=F (COMEX Copper Futures)
- Lithium: LIT (Global X Lithium & Battery Tech ETF)

### Appendix B: GARCH Parameter Estimates

**Baseline Models** (all commodities):
- ω (constant): 0.02-0.15
- α (ARCH term): 0.08-0.15
- β (GARCH term): 0.75-0.88

**Enhanced Models:**
- Same volatility parameters (ω, α, β)
- Mean equation adds β₁ (geo_keyword_hits) and β₂ (GPRD)
- None of β₁ or β₂ individually significant at α=0.05

### Appendix C: Hyperparameter Tuning Results

Grid search over:
- Sequence length: [10, 15, 20, 25]
- LSTM neurons: [16, 24, 32, 48]
- Learning rate: [0.0001, 0.0005, 0.001, 0.005]
- Batch size: [32, 64, 128]

Optimal (by validation loss):
- Sequence length: 15
- LSTM neurons: 32 (baseline), 24 (enhanced)
- Learning rate: 0.001
- Batch size: 64 (baseline), 32 (enhanced)

**Rationale:** With 21 inputs (vs. 6), reducing network size helps regularize and limit overfitting.

### Appendix D: Statistical Test Details

**Bootstrap procedure:**
1. Sample n commodities with replacement
2. Compute mean Sharpe improvement
3. Repeat 10,000 times
4. Take confidence intervals from the bootstrap distribution

**Results:** 90% CI: [-0.148, 0.235], Probability of positive effect: 54.0%

### Appendix E: Code Repository

**GitHub Repository:** https://github.com/Arttat69/capstone-data-science.git

### Appendix F: Feature Correlation Matrices

![Feature Correlations](data/output/appendix_plots/appendix_f_correlation_heatmaps.png)

**Figure F.1:** Strong correlation between GPRD and geo_keyword_hits (r = 0.65–0.78) helps explain joint significance with weak individual significance. Returns are weakly autocorrelated, while volatility is persistent.

### Appendix G: Volatility Dynamics During Geopolitical Events

![WTI Volatility](data/output/appendix_plots/appendix_g_volatility_wti.png)

**Figure G.1:** WTI volatility spikes often align with elevated GPRD in major crises, but many GPRD peaks do not translate into comparable volatility spikes.

![Wheat Volatility](data/output/appendix_plots/appendix_g_volatility_wheat.png)

**Figure G.2:** Wheat shows strong volatility response during Ukraine (2022), consistent with high estimated sensitivity.

![Gold Volatility](data/output/appendix_plots/appendix_g_volatility_gold.png)

**Figure G.3:** Gold spikes mainly during systemic crises; localized conflicts often have limited effect.

### Appendix H: Feature Distributions

![Feature Distributions](data/output/appendix_plots/appendix_h_feature_distributions.png)

**Figure H.1:** Returns are heavy-tailed; GPRD is right-skewed; volatility is positively skewed; geo_keyword_hits is sparse with occasional spikes.

### Appendix I: Time-Series Evolution of Key Variables

![WTI Time Series](data/output/appendix_plots/appendix_i_timeseries_wti.png)

**Figure I.1:** WTI price/volatility co-move with major crises; GPRD and news hits move closely together.

![Wheat Time Series](data/output/appendix_plots/appendix_i_timeseries_wheat.png)

**Figure I.2:** Wheat spikes in 2008, 2012, and 2022; the Ukraine episode coincides with GPRD elevation and large volatility increase.

![Copper Time Series](data/output/appendix_plots/appendix_i_timeseries_copper.png)

**Figure I.3:** Copper volatility aligns more with demand shocks than with GPRD peaks, consistent with weaker geopolitics channel.

### Appendix J: LSTM Architecture Diagrams

#### Baseline Model Architecture
```
Input: (batch_size, 15, 6)
↓
LSTM(32), return_sequences=True
↓
Dropout(0.2)
↓
LSTM(16)
↓
Dropout(0.2)
↓
Dense(1)
```
**Total parameters:** 6,657

#### Enhanced Model Architecture
```
Input: (batch_size, 15, 21)
↓
LSTM(24), return_sequences=True
↓
Dropout(0.2)
↓
LSTM(8)
↓
Dropout(0.2)
↓
Dense(1)
```
**Total parameters:** 5,969

**Design note:** Smaller enhanced models limit overfitting in higher-dimensional feature spaces.

### Appendix K: Event Study Summary Statistics

#### Pre/Post Event Volatility Changes

| Event                  | Commodity     | Pre-Event Vol_5 | Post-Event Vol_5 | % Change |
|------------------------|---------------|-----------------|------------------|----------|
| 2008 Financial Crisis  | WTI           | 0.024           | 0.041            | +71%     |
| 2008 Financial Crisis  | Gold          | 0.012           | 0.019            | +58%     |
| 2008 Financial Crisis  | Wheat         | 0.018           | 0.032            | +78%     |
| COVID-19 (March 2020)  | WTI           | 0.019           | 0.087            | +358%    |
| COVID-19 (March 2020)  | Gold          | 0.011           | 0.015            | +36%     |
| Ukraine (Feb 2022)     | WTI           | 0.016           | 0.028            | +75%     |
| Ukraine (Feb 2022)     | Wheat         | 0.021           | 0.039            | +86%     |
| Ukraine (Feb 2022)     | Natural Gas   | 0.034           | 0.052            | +53%     |

**Window:** Pre = 10 trading days before; Post = 10 trading days after.  
**Note:** Large post-shock jumps highlight why timing is difficult for daily forecasting.

### Appendix L: Likelihood Ratio Test Results (Complete Table)

| Commodity   | Model Specification           | LR Statistic | df | p-value  | Result      |
|-------------|-------------------------------|--------------|----|----------|-------------|
| Gold        | + geo_keyword_hits            | 90.82        | 1  | < 1e-21  | Highly Sig. |
| Gold        | + GPRD                        | 92.43        | 1  | < 1e-21  | Highly Sig. |
| Gold        | + geo_keyword_hits + GPRD     | 92.44        | 2  | < 1e-20  | Highly Sig. |
| WTI         | + geo_keyword_hits            | 119.38       | 1  | < 1e-27  | Highly Sig. |
| WTI         | + GPRD                        | 120.33       | 1  | < 1e-27  | Highly Sig. |
| WTI         | + geo_keyword_hits + GPRD     | 120.36       | 2  | < 1e-26  | Highly Sig. |
| Wheat       | + geo_keyword_hits            | 181.00       | 1  | < 1e-40  | Highly Sig. |
| Wheat       | + GPRD                        | 180.74       | 1  | < 1e-40  | Highly Sig. |
| Wheat       | + geo_keyword_hits + GPRD     | 181.01       | 2  | < 1e-40  | Highly Sig. |
| Natural Gas | + geo_keyword_hits            | 99.20        | 1  | < 1e-23  | Highly Sig. |
| Natural Gas | + GPRD                        | 99.50        | 1  | < 1e-23  | Highly Sig. |
| Natural Gas | + geo_keyword_hits + GPRD     | 99.58        | 2  | < 1e-22  | Highly Sig. |
| Copper      | + geo_keyword_hits            | 180.44       | 1  | < 1e-40  | Highly Sig. |
| Copper      | + GPRD                        | 179.15       | 1  | < 1e-40  | Highly Sig. |
| Copper      | + geo_keyword_hits + GPRD     | 180.57       | 2  | < 1e-40  | Highly Sig. |
| Lithium     | + geo_keyword_hits            | 129.09       | 1  | < 1e-29  | Highly Sig. |
| Lithium     | + GPRD                        | 126.89       | 1  | < 1e-29  | Highly Sig. |
| Lithium     | + geo_keyword_hits + GPRD     | 129.10       | 2  | < 1e-28  | Highly Sig. |

### Appendix M: LSTM Feature Set Comparison Summary

**Average Performance by Feature Set:**

| Feature Set | Mean Test R² | Mean RMSE | Mean Sharpe | Mean Alpha |
|-------------|--------------|-----------|-------------|------------|
| Baseline    | -0.215       | 0.0247    | 0.123       | -0.036     |
| GPRD        | -0.216       | 0.0247    | 0.117       | -0.033     |
| GEO         | -0.205       | 0.0247    | 0.143       | -0.007     |
| Combined    | -0.214       | 0.0247    | 0.145       | -0.014     |
| Granger     | -0.205       | 0.0247    | 0.124       | -0.035     |

**Best Feature Set by Commodity (by Sharpe Ratio):**

| Commodity   | Best Feature Set | Sharpe | Test R² | Alpha   |
|-------------|------------------|--------|---------|---------|
| Gold        | Baseline         | 0.681  | -0.061  | -0.001  |
| WTI         | Combined         | -0.360 | -0.015  | -0.000  |
| Wheat       | GPRD/GEO/Combined| 0.310  | -0.011  | -0.001  |
| Natural Gas | Granger          | 0.095  | -0.463  | +0.186  |
| Copper      | GEO              | 0.463  | -0.113  | -0.001  |
| Lithium     | Baseline/Combined| -0.052 | -0.628/-0.638 | -0.001  |

### Appendix N: GARCH Model AIC Comparison

**AIC by Model Specification:**

| Commodity   | Baseline | + geo | + GPRD | + geo + GPRD | Best Model | AIC Reduction | % Improvement |
|-------------|----------|-------|--------|--------------|------------|---------------|---------------|
| Gold        | 12668.50 | 12579.68 | 12578.07 | 12580.06 | + GPRD | 90.43 | 0.71% |
| WTI         | 19632.65 | 19515.28 | 19514.32 | 19516.29 | + GPRD | 118.33 | 0.60% |
| Wheat       | 18747.86 | 18568.86 | 18569.12 | 18570.85 | + geo | 179.00 | 0.95% |
| Natural Gas | 15213.63 | 15116.43 | 15116.13 | 15118.05 | + GPRD | 97.50 | 0.64% |
| Copper      | 16633.87 | 16455.43 | 16456.72 | 16457.30 | + geo | 178.44 | 1.07% |
| Lithium     | 9227.88  | 9100.78  | 9102.99  | 9102.78  | + geo | 127.09 | 1.38% |

**Interpretation:** All commodities show meaningful AIC improvements when geopolitical variables are added (0.60%–1.38%). Since AIC penalizes additional parameters, these improvements indicate genuine explanatory value. The ranking reveals heterogeneity: industrial metals and agriculture show stronger associations than energy commodities.