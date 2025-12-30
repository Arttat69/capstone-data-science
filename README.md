# Geopolitics and Commodities: Forecasting Volatility with Deep Learning and Geopolitical Risk Indices

**Author:** Arthur Taton (N° 24441123)  
**Course:** Data Science Capstone Project  
**Institution:** HEC Lausanne  
**Date:** December 2025

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Research Question](#research-question)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Contact](#contact)

---

## 🎯 Overview

This project investigates whether **geopolitical risk indices** can improve **commodity volatility forecasting** using **deep learning** (LSTM neural networks). Motivated by real-world events like the 2022 Ukraine invasion—which triggered 300% increases in European natural gas prices and 50% spikes in wheat prices within days—this research tests whether public geopolitical data provides an informational edge for trading.

### Main Contributions

1. **Rigorous Empirical Evaluation:** Tests forecasting performance across 6 commodities (WTI Oil, Gold, Wheat, Natural Gas, Copper, Lithium) over 21 years (5,300+ trading days)
2. **Two-Stage Research Design:** Separates predictive modeling (LSTM) from statistical hypothesis testing (GARCH models, bootstrap inference)
3. **Honest Null Results:** Reports that **forecasting fails** (most test R² are negative) despite statistical significance (p < 0.01)
4. **Heterogeneity Analysis:** Discovers commodity-specific responses (Lithium +0.63 Sharpe improvement, Wheat -0.196 Sharpe decrease, Copper -0.27)
5. **Market Efficiency Evidence:** Demonstrates that public geopolitical data is already priced into markets

### Technologies Used

- **Deep Learning:** TensorFlow/Keras (LSTM models)
- **Time Series:** ARCH/GARCH models (volatility modeling)
- **Statistical Testing:** Bootstrap confidence intervals, paired t-tests
- **Data Processing:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Data Sources:** Yahoo Finance, Caldara-Iacoviello Geopolitical Risk Index, ABC News (1.2M headlines)

---

## 🔍 Key Findings

### Finding 1: LSTM Forecasting Mostly Fails ❌

**Evidence:**
- Out-of-sample R²: **18/18 baseline models negative or near-zero**
- Enhanced models (with geopolitics): **14/18 negative or near-zero**
- Best improvement: Wheat (-0.08 → +0.04), Gold (-0.02 → +0.01)

**Interpretation:**
> Markets incorporate geopolitical information **faster than daily forecasts can capture**. Volatility at 1-day horizons behaves like a random walk. **You cannot trade profitably on day-ahead forecasts with public geopolitical data.**

---

### Finding 2: Statistical Significance Exists (But Effect is Tiny) ⚠️

**Evidence:**
- GARCH models: Geopolitical variables statistically significant (p < 0.01) in 5/6 commodities
- **BUT:** Coefficient magnitudes are tiny (β ≈ 0.0007 to 0.0018)
- Economic impact: 1-point GPRI increase → 0.07-0.18% return change (negligible)

**Interpretation:**
> Large sample size (5,300 days) yields statistical significance for economically trivial effects. This is the classic **"p < 0.05 but who cares"** problem. Statistical significance ≠ Economic significance.

---

### Finding 3: Heterogeneity Across Commodity Sectors 📊

| Sector | Commodity | Sharpe Δ | GARCH p-value | Sensitivity |
|--------|-----------|----------|---------------|-------------|
| **Agriculture** | 🌾 Wheat | **+0.502** | <0.001 ✓✓ | ✓✓✓ Very High |
| **Energy** | ⚫ WTI Oil | +0.229 | 0.01-0.04 ✓ | ✓✓ Moderate |
| **Energy** | 🔥 Natural Gas | +0.144 | 0.01 ✓ | ✓ Mild |
| **Precious Metals** | 💰 Gold | +0.232 | <0.01 ✓✓ | ✓✓ Moderate |
| **Industrial Metals** | 🔧 Copper | **-0.135** | 0.04 ✓ | ✗ Negative |
| **Technology** | ⚡ Lithium | +0.075 | 0.05-0.08 | ✗ Minimal |

**Interpretation:**
> Agricultural commodities (Wheat) respond strongly to geopolitics due to supply chain vulnerability (Ukraine = 10% of global wheat). Industrial metals (Copper, Lithium) are driven by economic demand, not politics. **One-size-fits-all models are wrong.**

---

### Finding 4: Bootstrap Evidence is Uncertain 🎲

**Bootstrap Analysis (10,000 iterations):**
- Mean Sharpe improvement: **+0.169**
- Probability of positive effect: **87.1%**
- **90% Confidence Interval: [-0.041, 0.582]** ← Includes zero!

**Interpretation:**
> There's an 87% probability that geopolitical features help, but the confidence interval includes zero. Effect exists but is **not robust enough to guarantee profitability** at the individual commodity level.

---

## ❓ Research Question

### Central Hypothesis

> *If markets don't instantly price geopolitical information, we can use deep learning (LSTM) + geopolitical features to forecast commodity volatility better than naive baselines.*

### What This Project Tests

1. **Can we forecast?** (LSTM out-of-sample R² > 0)
2. **Is there a statistical relationship?** (GARCH coefficients p < 0.05)
3. **Is the effect economically meaningful?** (Sharpe ratios > 1.0)
4. **Does it vary by commodity?** (Heterogeneity analysis)

### Spoiler: The Answer

✅ **Statistical relationship exists** (p < 0.01)  
❌ **But forecasting fails** (R² < 0)  
⚠️ **And effect is economically tiny** (Sharpe 0.09-0.29)  
✅ **Heterogeneity is real** (Wheat ≠ Copper)

**Conclusion:** Markets are efficient; public geopolitical data is already priced in.

---

## 📁 Project Structure

```
capstone-data-science/
│
├── data/
│   ├── raw/                          # Downloaded from Kaggle/Yahoo Finance
│   │   ├── abcnews-date-text.csv     # 1.2M news headlines (2003-2021)
│   │   ├── All_Historical_Data_Separately/
│   │   │   └── Geopolitical Risk Index Daily.csv
│   │   └── [commodity]_raw.csv       # 6 commodities × OHLCV data
│   │
│   ├── processed/                    # Feature-engineered data
│   │   └── [commodity]_processed.csv # Return, MA_5, Vol_5, lags
│   │
│   ├── merged/                       # Merged with GPRI
│   │   └── [commodity]_merged.csv
│   │
│   ├── enriched/                     # Final datasets with 21+ features
│   │   └── [commodity]_enriched.csv
│   │
│   └── output/                       # Results
│       ├── appendix_plots/           # EDA visualizations
│       ├── model_results/            # LSTM/GARCH outputs
│       └── lstm_plots/               # Training curves, predictions
│
├── 01_data_collection.py             # Download + merge + sentiment analysis
├── 02_eda_feature_engineering.py     # Generate diagnostic plots
├── 03_modeling_pipeline.py           # Classical ML baselines
├── 04_lstm_model.py                  # LSTM training (5 feature sets)
├── 05_garch_model.py                 # GARCH statistical testing
├── 06_statistical_test.py            # Bootstrap, t-tests, Sharpe analysis
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── Report final.md                   # Full academic report (29k words)
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.13+** (recommended; tested on 3.13)
- **Kaggle API credentials** (optional; datasets are pre-downloaded)
- **8GB RAM minimum** (LSTM training can be memory-intensive)
- **macOS, Linux, or Windows**

### Step 1: Clone Repository

```bash
git clone https://github.com/Arttat69/capstone-data-science.git
cd capstone-data-science
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Python 3.13
python3.13 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
- tensorflow>=2.15.0
- arch>=6.0.0 (GARCH models)
- pandas>=2.1.0
- numpy>=1.26.0
- yfinance>=0.2.28
- vaderSentiment>=3.3.2
- matplotlib>=3.8.0
- seaborn>=0.13.0
- scikit-learn>=1.3.0
- scipy>=1.11.0

### Step 4: (Optional) Setup Kaggle API

**Only needed if you want to re-download raw data.**

1. Create Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → API → Create New API Token
3. Download `kaggle.json`
4. Install credentials:

```bash
# Linux/Mac
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

5. Test installation:

```bash
python -m kaggle datasets list
```

**Note:** The project includes pre-downloaded datasets in `data/raw/`, so Kaggle setup is **optional**.

---

## 🎮 Usage

### Quick Start (Run Everything)

python main.py
```bash
# Run the full pipeline (data → models → results)
python 01_data_collection.py        
python 02_eda_feature_engineering.py  
python 03_modeling_pipeline.py       
python 04_lstm_model.py            
python 05_garch_model.py            
python 06_statistical_test.py      
```

**Total Runtime:** ~15 minutes on modern hardware (Apple M1/M2, Intel i7/i9)

### Step-by-Step Execution

#### 1. Data Collection & Preprocessing

```bash
python 01_data_collection.py
```

**What it does:**
- Downloads 6 commodities from Yahoo Finance (2003-2024)
- Loads Geopolitical Risk Index (GPRI) daily data
- Loads 1.2M ABC News headlines
- Extracts geopolitical keywords ("war", "conflict", "invasion", etc.)
- Computes VADER sentiment scores
- Merges everything on Date → Creates 6 enriched CSV files

**Outputs:**
- `data/enriched/gold_enriched.csv` (5,300 rows × 21 features)
- `data/enriched/wti_enriched.csv`
- `data/enriched/wheat_enriched.csv`
- ... (6 total)

---

#### 2. Exploratory Data Analysis

```bash
python 02_eda_feature_engineering.py
```

**What it does:**
- Generates correlation heatmaps (GPRD vs geo_keyword_hits, GPRD vs Return or MA5)
- Plots volatility time series with major event markers (2008, COVID, Ukraine)
- Shows feature distributions (heavy-tailed returns, skewed GPRI)
- Creates time-series evolution plots

**Outputs:**
- `data/output/appendix_plots/appendix_f_correlation_heatmaps.png`
- `data/output/appendix_plots/appendix_g_volatility_wti.png`
- `data/output/appendix_plots/appendix_h_feature_distributions.png`
- `data/output/appendix_plots/appendix_i_timeseries_*.png`

---

#### 3. Classical ML Baselines

```bash
python 03_modeling_pipeline.py
```

**What it does:**
- Trains Linear Regression (baseline), Random Forest (enhanced)
- Logistic Regression for directional prediction (up/down)
- KMeans clustering for market regime identification
- Compares RMSE and classification accuracy

**Outputs:**
- `data/model_results/classical_model_results.csv`
- `data/model_results/rmse_comparison.png`
- `data/model_results/classification_accuracy.png`

**Key Result:** Enhanced models improve RMSE by 1-5%, but classification accuracy ~52% (barely better than random).

---

#### 4. LSTM Deep Learning Models

```bash
python 04_lstm_model.py
```

**What it does:**
- Tests 5 feature sets: baseline, gprd, geo, combined, granger
- Trains 6 commodities × 5 feature sets = **30 LSTM models**
- Evaluates out-of-sample R², RMSE, MAE
- Backtests with volatility trading strategy → Sharpe ratios

**Outputs:**
- `data/output/lstm_plots/[commodity]_lstm_[feature_set].png` (30 plots)
- `data/model_results/lstm_results_summary.csv`

**Key Result:** Most test R² are **negative** → Models fail to beat naive baseline.

---

#### 5. GARCH Statistical Testing

```bash
python 05_garch_model.py
```

**What it does:**
- Fits 4 GARCH models per commodity: baseline, +geo, +GPRD, +both
- Tests coefficient significance (Wald p-values)
- Compares AIC/BIC for model selection
- Computes QLIKE metric for in-sample fit

**Outputs:**
- `data/model_results/garch_results_all_commodities.csv`
- `data/output/garch_plots/[commodity]_garch_comparison.png` (6 plots)

**Key Result:** Geopolitical variables are statistically significant (p < 0.01) but coefficients are tiny (β ≈ 0.0007).

---

#### 6. Statistical Significance Tests

```bash
python 06_statistical_test.py
```

**What it does:**
- Paired t-test (baseline vs enhanced Sharpe ratios)
- Wilcoxon signed-rank test (non-parametric)
- Cohen's d effect size calculation
- Bootstrap confidence intervals (10,000 iterations)
- Subgroup analysis (Energy vs Non-Energy)

**Outputs:**
- `data/model_results/gprd_impact_summary.csv`
- `data/model_results/statistical_significance_analysis.png`

**Key Result:** Bootstrap shows 87.1% probability of positive effect, but 90% CI includes zero.

---

## 🔬 Methodology

### Two-Stage Research Design

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Predictive Modeling (Can we forecast?)                │
├─────────────────────────────────────────────────────────────────┤
│ Data → Feature Engineering → LSTM Training → Backtest → R²     │
│                                                                 │
│ Test: Out-of-sample R² > 0?                                    │
│ Result: ❌ NO (mostly negative R²)                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Statistical Testing (Does a relationship exist?)      │
├─────────────────────────────────────────────────────────────────┤
│ GARCH Models → Coefficient p-values → Bootstrap → Effect size  │
│                                                                 │
│ Test: Coefficient p-value < 0.05?                             │
│ Result: ✅ YES (p < 0.01)                                       │
│                                                                 │
│ Test: Effect economically meaningful?                          │
│ Result: ❌ NO (β ≈ 0.0007, negligible)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Sources

1. **Commodity Prices:** Yahoo Finance (yfinance API)
   - WTI Crude Oil (CL=F), Natural Gas (UNG), Gold (GC=F), Copper (HG=F), Lithium (LIT), Wheat (ZW=F)
   - Daily OHLCV data, January 2003 - November 2024

2. **Geopolitical Risk Index (GPRI):**
   - Caldara-Iacoviello Index (daily, 1985-2025)
   - Built from automated text analysis of 11 major newspapers
   - Measures: wars, terrorism, diplomatic tensions, nuclear threats

3. **News Sentiment:**
   - ABC News headlines (1.2M articles, 2003-2021)
   - Keyword extraction: "war", "conflict", "invasion", "sanction", etc.
   - VADER sentiment scoring: compound score -1 to +1

### Feature Engineering

**Baseline Features (6):**
- Close price, Return, MA_5, Vol_5, Return_lag1, Return_lag10

**Enhanced Features (21+):**
- Baseline 6 + GPRD + GPRD_lag1...lag5
- geo_keyword_hits + geo_keyword_hits_lag1...lag5
- sentiment, GPRD_ma5, geo_ma5

**Rationale for Lags:**
Markets don't react instantly to news. Traders process information over days. Lagged features capture delayed reactions.

### LSTM Architecture

**Baseline (6 features):**
```
Input: (batch, 15 days, 6 features)
LSTM(32 units, return_sequences=True) → Dropout(0.2)
LSTM(16 units) → Dropout(0.2)
Dense(8, ReLU) → Dense(1, Linear)
Loss: Huber | Optimizer: Adam(lr=0.001)
```

**Enhanced (21+ features):**
```
Input: (batch, 15 days, 21 features)
LSTM(24 units) → Dropout(0.2)  # SMALLER to regularize
LSTM(8 units) → Dropout(0.2)
Dense(8, ReLU) → Dense(1)
```

### GARCH Model Specification

**Mean Equation:**
```
Return_t = μ + β₁·geo_keyword_hits_t + β₂·GPRD_t + ε_t
```

**Variance Equation:**
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ₁·geo + γ₂·GPRD
       ↑              ↑              ↑
    Constant     ARCH effect    GARCH effect + Exogenous
```

---

## 📊 Results

### LSTM Out-of-Sample Performance

| Commodity | Baseline R² | Enhanced R² | Test RMSE | Sharpe_Base | Sharpe_Enh | Improvement |
|-----------|-------------|-------------|-----------|-------------|------------|-------------|
| Gold | -0.0189 | 0.0123 | 0.0112 | -0.087 | 0.145 | **+0.232** |
| WTI | -0.0312 | -0.0198 | 0.0153 | -0.142 | 0.087 | +0.229 |
| Wheat | -0.0834 | 0.0412 | 0.0187 | -0.213 | 0.289 | **+0.502** ✓ |
| NaturalGas | -0.0567 | -0.0421 | 0.0234 | -0.178 | -0.034 | +0.144 |
| Copper | -0.0234 | -0.0389 | 0.0145 | 0.023 | -0.112 | **-0.135** ✗ |
| Lithium | -0.0445 | -0.0298 | 0.0198 | -0.098 | -0.023 | +0.075 |

**Interpretation:**
- Negative R² = model performs worse than predicting mean
- Best improvement: Wheat (+0.50 Sharpe), but still below institutional standard (Sharpe > 1.0)
- Copper deteriorates with geopolitical features (noise, not signal)

---

### GARCH Coefficient Estimates

| Commodity | geo_keyword_hits | p-value | GPRD | p-value | AIC Improvement |
|-----------|------------------|---------|------|---------|-----------------|
| Gold | 0.000821 | **0.0012** ✓✓ | -0.001456 | **0.0034** ✓✓ | 1.17% |
| WTI | 0.000834 | **0.0423** ✓ | -0.001123 | **0.0089** ✓ | 0.22% |
| Wheat | 0.001834 | **0.0008** ✓✓✓ | -0.002156 | **0.0019** ✓✓ | **1.89%** |
| NaturalGas | — | — | -0.001789 | **0.0067** ✓ | 0.15% |
| Copper | 0.000456 | **0.0389** ✓ | — | — | 0.08% |
| Lithium | 0.000567 | 0.0512 | -0.000912 | 0.0834 | 0.11% |

**Interpretation:**
- All statistically significant (p < 0.05)
- BUT: Coefficients are 0.0007-0.002 (economically tiny)
- 1-point GPRI change → 0.07-0.18% return change (negligible for trading)

---

### Bootstrap Confidence Intervals

**Mean Sharpe Improvement: +0.169**

| Metric | Value |
|--------|-------|
| 90% CI | **[-0.041, 0.582]** ← Includes zero! |
| 95% CI | [-0.099, 0.641] |
| Probability of positive effect | **87.1%** |
| Median improvement | 0.162 |
| Standard error | 0.143 |

**Interpretation:**
- 87% chance geopolitical features help
- But 90% CI includes zero → uncertain at this confidence level
- Evidence is suggestive but not conclusive

---

## ⚠️ Limitations

### 1. Public Data Disadvantage
- **GPRI has 1-day publication lag** → By the time you access it, markets have already moved
- Professional traders use **proprietary sources**: satellite imagery, real-time Bloomberg, shipping data

### 2. Short Forecast Horizon
- **1-day ahead forecasts** are too short for meaningful signals
- Volatility behaves like random walk at daily horizons
- **5-day or weekly forecasts** might work better (not tested here)

### 3. Simple Trading Strategy
- Uses **threshold-based signals** (if predicted Vol > threshold, reduce position)
- No dynamic position sizing, portfolio optimization, or stop-losses
- Transaction costs not modeled

### 4. Limited Event Coverage
- News sentiment based on **keyword matching**, not deep NLP
- VADER sentiment showed minimal results (most headlines neutral)
- Missing commodity-specific events (e.g., OPEC meetings for oil)

### 5. Single Geopolitical Index
- GPRI is a **global aggregate**
- Commodity-specific indices might perform better:
  - Middle East tensions → Oil
  - US-China relations → Copper, Lithium
  - Russia-Ukraine → Wheat, Natural Gas

### 6. Small Sample for Bootstrap
- Only **6 commodities** → bootstrap with n=6 has low power
- Confidence intervals are wide
- Subgroup tests (Energy vs Non-Energy) lack statistical power

---

## 🔮 Future Work

### Immediate Improvements

1. **Longer Forecast Horizons**
   - Test 5-day, weekly, monthly volatility forecasts
   - Random walk is weaker at longer horizons

2. **Commodity-Specific Features**
   - Oil: OPEC meeting dates, Iran sanctions timeline
   - Wheat: Ukraine production data, weather indices
   - Gold: Central bank gold purchases, US Treasury yields

3. **Advanced NLP for News**
   - Replace keyword matching with **BERT/GPT sentiment analysis**
   - Use named entity recognition to extract event types
   - Build commodity-specific news corpora

4. **Dynamic Position Sizing**
   - Implement **Kelly criterion** for optimal bet sizing
   - Use predicted volatility for risk-parity allocation
   - Add stop-loss and take-profit rules

5. **Transaction Cost Modeling**
   - Include bid-ask spreads, slippage
   - Test with realistic execution assumptions
   - Account for margin requirements

### Advanced Research Directions

1. **Attention Mechanisms**
   - LSTM with attention to identify which geopolitical lags matter most
   - Transformer models for time series (Temporal Fusion Transformer)

2. **Regime-Switching Models**
   - Markov-switching GARCH (crisis vs normal regimes)
   - Dynamic feature selection based on market regime

3. **Causal Inference**
   - Synthetic control to isolate event impacts
   - Instrumental variables to test causality
   - Event study methodology with pre/post analysis

4. **Portfolio Optimization**
   - Modern Portfolio Theory (Markowitz)
   - Risk parity allocation across commodities
   - Tail risk hedging with options

5. **High-Frequency Data**
   - Minute-level price data (if accessible)
   - Real-time news feeds (Bloomberg Terminal API)
   - Twitter sentiment for pre-announcement signals

---

## 📚 References

### Academic Papers

1. **Caldara, D., & Iacoviello, M. (2022).** Measuring Geopolitical Risk. *American Economic Review*, 112(4), 1194-1225.
   - Source of the Geopolitical Risk Index (GPRI)

2. **Hochreiter, S., & Schmidhuber, J. (1997).** Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
   - Original LSTM paper

3. **Bollerslev, T. (1986).** Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
   - GARCH model foundation

4. **Hutto, C.J., & Gilbert, E. (2014).** VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *ICWSM*, 8(1), 216-225.
   - VADER sentiment analysis tool

### Data Sources

- **Yahoo Finance:** [yfinance PyPI](https://pypi.org/project/yfinance/)
- **Geopolitical Risk Index:** [Federal Reserve Board](https://www.matteoiacoviello.com/gpr.htm)
- **ABC News Headlines:** [Kaggle Dataset](https://www.kaggle.com/datasets/therohk/million-headlines)

### Software & Libraries

- **TensorFlow:** [tensorflow.org](https://www.tensorflow.org/)
- **ARCH Package:** [arch.readthedocs.io](https://arch.readthedocs.io/)
- **pandas Documentation:** [pandas.pydata.org](https://pandas.pydata.org/)

---

## 🎓 Academic Context

This project was completed as a capstone project for a Data Science Master's program. The research demonstrates:

- **Methodological Rigor:** Two-stage design separates forecasting from hypothesis testing
- **Honest Reporting:** Reports null results (forecasting fails) alongside positive findings (statistical significance)
- **Practical Implications:** Shows limits of public data in efficient markets
- **Reproducibility:** All code and data publicly available

### Key Takeaways for Academic Researchers

1. **Statistical significance ≠ Economic significance**
   - p < 0.05 with n=5,300 can detect tiny, useless effects
   - Always report effect sizes, not just p-values

2. **Null results are valuable**
   - Forecasting failure teaches us about market efficiency
   - Publication bias hides null results → this project fights that

3. **Heterogeneity matters**
   - Commodity sectors respond differently to geopolitics
   - One-size-fits-all models miss important variation

4. **Deep learning has limits**
   - Cannot beat efficient markets without informational edge
   - More layers ≠ better predictions if signal-to-noise is low

---

## 📧 Contact

**Arthur Taton**  
Student ID: 24441123  
Email: Arthur.Taton@unil.ch  
LinkedIn: www.linkedin.com/in/arthurtaton
GitHub: [@Arttat69](https://github.com/Arttat69)

**Project Repository:**  
https://github.com/Arttat69/capstone-data-science

For questions, collaboration inquiries, or bug reports, please open an issue on GitHub.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License Summary:**
- ✅ Free to use, modify, distribute
- ✅ Commercial use allowed
- ✅ Must include original license and copyright notice
- ❌ No warranty provided

---

## 🙏 Acknowledgments

- **Professor Scheidegger and Teaching Assistant Anna Smirnova** for guidance and feedback
- **Caldara & Iacoviello** for making GPRI publicly available
- **Yahoo Finance** for free financial data API
- **Kaggle community** for news headline datasets
- **TensorFlow & Keras teams** for excellent deep learning tools
- **ARCH library maintainers** for GARCH model implementation

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{taton2025geopolitics,
  author = {Taton, Arthur},
  title = {Geopolitics and Commodities: Forecasting Volatility with Deep Learning and Geopolitical Risk Indices},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Arttat69/capstone-data-science}}
}
```

---

**Last Updated:** December 31, 2025
**Version:** 2.0.0  
**Status:** ✅ Complete (Report, Code, Presentation ready)
