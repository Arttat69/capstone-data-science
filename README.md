# capstone-data-science
Capstone project for Data Science course: stock market prediction with machine learning
# CapstoTitle:
Geopolitics and Commodities: Predicting Oil, Gold, and Wheat Price Movements with Machine Learning

Abstract:
Commodity markets are highly sensitive to geopolitical developments. Wars, sanctions, and policy decisions often trigger significant shifts in the prices of oil, gold, and agricultural goods. This project investigates whether machine learning methods can capture and predict these dynamics by combining financial time series with geopolitical context.
Using historical data for oil, gold, and wheat, the project applies a variety of techniques learned in the course. Regression models are used to forecast short-term price changes, while classification algorithms predict whether commodities will rise or fall following key events. Unsupervised clustering methods explore whether commodities form distinct groups (e.g., “safe havens” vs. “volatile assets”) during crises. In addition, a simple neural network is implemented to compare deep learning performance with classical models.
The analysis involves extensive data cleaning, exploratory data analysis, and feature engineering, including the creation of lagged returns and event-based indicators. Performance is evaluated across models using standard metrics, and visualizations highlight how geopolitical shocks propagate through commodity markets. Finally, the project reflects on the limits of prediction in politically driven environments and discusses potential improvements with larger datasets or text-based event sentiment.
By combining economics, finance, and political science, this capstone project illustrates both the power and the boundaries of machine learning in understanding complex real-world markets.

📊 Project Plan

Title: Geopolitics and Commodities: Predicting Oil, Gold, and Wheat Price Movements with Machine Learning

1. 🎯 Objective
To investigate how geopolitical risk and global events affect commodity prices (Oil, Gold, Wheat) and to build predictive models (regression, classification, deep learning) that forecast price movements by incorporating both price history and geopolitical indicators.

2. 📂 Data Sources
Commodity Prices

Yahoo Finance (yfinance) → Oil (CL=F), Gold (GC=F), Wheat (ZW=F).

Kaggle datasets:

Gold-Silver Price vs Geopolitical Risk (1985–2025)

Crude Oil Prices Dataset

Wheat Futures Prices

Geopolitical Data

Kaggle Global News Dataset (for event/sentiment features).

Geopolitical Risk Index (from Kaggle dataset above).

Optionally: manually defined major geopolitical shocks (wars, sanctions, OPEC announcements).

3. 🛠 Methodology
Step 1: Data Collection & Preprocessing

Download commodity price data (daily).

Align with geopolitical risk index (monthly → resample to daily/weekly).

Clean and preprocess news data:

Text cleaning (optional NLP sentiment analysis).

Simpler feature: count headlines with keywords (e.g., “war”, “sanctions”, “conflict”).

Merge datasets on date.

Step 2: Exploratory Data Analysis (EDA)

Plot commodity prices over time.

Highlight major geopolitical events on price timelines.

Correlation between geopolitical risk index and price movements.

Distribution of returns during crisis vs non-crisis periods.

Step 3: Feature Engineering

Price returns (% daily change).

Rolling averages, volatility (5-day, 30-day).

Lag features (past returns).

Geopolitical indicators:

Geopolitical Risk Index.

News headline count (geopolitical keywords).

Event dummy variable (1 on event dates, 0 otherwise).

Step 4: Modeling
Use a progression of models (to demonstrate course learning):

Regression Models

Linear Regression (baseline).

Ridge/Lasso regression.

Classification Models

Logistic Regression (predict Up/Down).

Decision Tree / Random Forest.

k-Nearest Neighbors (for direction prediction).

Unsupervised Learning

k-Means clustering → detect “crisis vs normal” market regimes.

PCA → reduce dimensionality, visualize patterns.

Deep Learning

LSTM Neural Network → predict future price based on sequences of past prices + geopolitical features.

Step 5: Evaluation

Metrics:

Regression: RMSE, MAE.

Classification: Accuracy, Precision, Recall, F1, ROC-AUC.

Compare baseline (price-only) vs enriched models (with geopolitical features).

Interpret feature importance (tree-based models).

Step 6: Results & Interpretation

How strongly do geopolitical factors improve predictions?

Which commodities are most geopolitically sensitive (oil vs gold vs wheat)?

Visualizations:

Time series with event annotations.

Feature importance charts.

Model comparison tables.

Step 7: Deliverables

Report (~10 pages)

Introduction (motivation: geopolitics + commodities).

Literature review (brief — geopolitics & financial markets).

Data description.

Methodology.

Results & interpretation.

Conclusion & limitations.

GitHub Repository

data/ (raw + processed).

notebooks/ (EDA, modeling, results).

src/ (Python scripts for data loading, feature engineering, models).

README.md (overview + instructions).

Video (10 min)

Problem statement.

Dataset & features.

Methodology.

Key findings & visuals.

Limitations & next steps.

4. 📅 Timeline

Week 7–8 (Now): Data collection & cleaning. Start GitHub repo structure.

Week 9: EDA + Feature Engineering.

Week 10: Regression & Classification models.

Week 11: Unsupervised learning + start LSTM.

Week 12: Refine models, evaluation, visuals.

Week 13: Write report draft + record video.

Week 14: Finalize report, polish GitHub repo, submit.

5. 💡 Bonus Ideas for Extra Points

Add sentiment analysis from global news dataset using NLP (VADER or TextBlob).

Use High-Performance Computing concepts (parallelize feature extraction or model training).

Test models during different periods (e.g., stable vs crisis) → regime comparison.