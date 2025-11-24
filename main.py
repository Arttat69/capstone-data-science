# main.py

import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import data_collection  # from 01
import eda_feature_engineering  # from 02
import modeling_pipeline  # from 03
import lstm_model  # from 04
import statistical_test  # from 05

if __name__ == "__main__":
    print("Running data collection...")
    data_collection.main()

    print("Running EDA and feature engineering...")
    eda_feature_engineering.main()

    print("Running modeling pipeline...")
    modeling_pipeline.main()

    print("Running LSTM model...")
    lstm_model.main()

    print("Running statistical tests...")
    statistical_test.main()

    print("Pipeline complete.")