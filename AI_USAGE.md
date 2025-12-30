# AI Tool Usage Disclosure

## Tools Used
- **Claude AI (Anthropic)**: Primary assistant for debugging, code optimization, and architectural design
- **Grok**: Code completion and results analysis
- **ChatGPT**: Secondary consultation for statistical methodology questions

## Development Process

This project was developed through iterative collaboration with AI tools, with all core logic, feature engineering decisions, and research direction determined independently. AI assistance was primarily used for implementation efficiency and debugging rather than conceptual design.

## Significant Contributions by Component

### 1. Data Collection Pipeline (notebooks/01_data_collection.ipynb)
**AI Contribution:**
- Suggested yfinance API syntax for multi-ticker downloads
- Provided pandas merge strategies for time series alignment
- Generated template code for GPRD daily resampling

**My Work:**
- Designed overall data architecture and file structure
- Determined which commodities to include based on geopolitical relevance
- Implemented custom geopolitical keyword extraction logic
- Debugged date alignment issues.
- Created sentiment scoring methodology for news data

### 2. Feature Engineering
**AI Contribution:**
- Recommended standard technical indicators (moving averages, volatility)
- Suggested lag depth experimentation (1-5 days)
- Provided sklearn scaler implementation examples

**My Work:**
- Conceptualized GPRD-derived features (high-risk regime, moving average, shock detection)
- Determined optimal lag structure (3 days) through empirical testing
- Designed hierarchical feature set (baseline vs enhanced)
- Manually validated feature distributions and handled outliers
- Created commodity-specific feature selection logic

### 3. LSTM Model Architecture (notebooks/03_modeling_pipeline.ipynb)
**AI Contribution:**
- Suggested Huber loss for volatility outlier robustness
- Recommended early stopping and learning rate reduction callbacks
- Helped diagnose memory allocation errors during initial runs

**My Work:**
- Researched LSTM implementation 
- Determined final network architecture (32→16 for baseline, 24→8 for high-dimensional input)
- Tuned hyperparameters: sequence length (15), batch size (32/64), regularization strength
- Implemented dual-model experimental design (with/without GPRD)
- Created custom sequence generation function for time series windowing
- Developed trading signal logic from volatility predictions

### 4. Evaluation Framework
**AI Contribution:**
- Provided formulas for Sharpe ratio and maximum drawdown calculations
- Suggested statistical significance testing approach (paired t-tests)
- Generated matplotlib visualization templates

**My Work:**
- Designed comprehensive evaluation metric suite (R², Sharpe, alpha, drawdown)
- Implemented custom backtesting logic with transaction costs
- Created commodity comparison framework
- Interpreted results and identified heterogeneous GPRD impact pattern
- Generated all final visualizations with domain-specific annotations

### 5. Code Optimization and Debugging
**AI Contribution:**
- Diagnosed DataFrame indexing errors causing shape mismatches
- Suggested RobustScaler instead of StandardScaler for outlier handling
- Identified memory bottleneck from excessive feature lagging
- Provided error handling patterns for plotting edge cases

**My Work:**
- Refactored initial implementation to reduce memory usage by 57%
- Optimized feature generation to avoid redundant calculations
- Implemented data validation checks throughout pipeline
- Structured code for reproducibility and modularity

## Learning Moments

### Technical Skills Acquired
1. **LSTM Sequence Processing**: Used AI explanations to understand time series windowing and the difference between return_sequences=True/False. Implemented custom sequence creation logic after understanding the concept.

2. **Volatility Forecasting**: AI provided theoretical background on why volatility is harder to predict than returns. This informed my decision to evaluate models on trading metrics rather than just R².

3. **Feature Scaling Impact**: Learned through AI guidance that commodity prices have vastly different scales (oil vs gold), requiring robust normalization. Tested multiple scaler types before selecting RobustScaler.

4. **Memory Management in Deep Learning**: Debugged memory errors with AI assistance, learning that feature count × sequence length × batch size determines memory footprint. Applied this to reduce dimensionality strategically.

### Research Insights
1. **Geopolitical Risk Heterogeneity**: While AI suggested testing GPRD features, the discovery that energy commodities benefit (+127% Sharpe for WTI) while industrial metals suffer (-108% for Copper) emerged from my data analysis.

2. **Baseline Model Collapse**: AI did not anticipate that LSTM without external features would fail (negative R²). This finding—that deep learning requires domain knowledge—was my empirical contribution.

3. **Trading vs Forecasting Metrics**: AI provided standard ML metrics, but I independently recognized that low R² with high Sharpe ratio is acceptable for trading applications. This insight guided evaluation framework design.

## Verification and Validation

All code output was manually verified:
- Cross-checked Sharpe ratio calculations against finance literature formulas
- Validated GPRD merge alignment by spot-checking dates against source data
- Confirmed LSTM predictions by plotting actual vs predicted volatility
- Reproduced key results across multiple runs to ensure reproducibility

Statistical significance tests (paired t-tests) were implemented independently after consulting statistical methodology textbooks, with AI only providing scipy syntax.

## Ethical Considerations

This disclosure aims to transparently communicate AI's role in development while emphasizing that:
1. All research questions and hypotheses were self-generated
2. Feature engineering creativity (GPRD regime detection, shock indicators) was original
3. Key findings (heterogeneous impact, baseline failure) were not AI-suggested
4. Model architecture decisions were made through empirical iteration, not AI recommendation
5. All written analysis and interpretation is original work

The final codebase represents my understanding of machine learning principles applied to financial forecasting, with AI serving as an efficiency tool rather than a conceptual substitute.

## Code Ownership Statement

I certify that:
- I understand every line of code in this repository
- I can explain the mathematical basis for all models implemented
- All analytical conclusions are based on my interpretation of results
- This project represents my independent work with AI used as a development accelerator

The use of AI tools does not diminish the intellectual contribution required to:
- Formulate a novel research question about geopolitical risk heterogeneity
- Design an experimental framework comparing models across 6 commodities
- Interpret results to identify commodity class patterns
- Recognize that traditional ML metrics (R²) poorly measure trading strategy quality

---

**Last Updated**: December 2025