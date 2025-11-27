# main.py
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import data_collection  # from notebook 01
import eda_feature_engineering  # from notebook 02
import modeling_pipeline  # from notebook 03
import lstm_model  # from notebook 04
import statistical_test  # from notebook 05

if __name__ == "__main__":
    print("=" * 50)
    print("Starting ML Pipeline")
    print("=" * 50)

    print("\n[1/5] Running data collection...")
    data_collection.main()

    print("\n[2/5] Running EDA and feature engineering...")
    eda_feature_engineering.main()

    print("\n[3/5] Running modeling pipeline...")
    modeling_pipeline.main()

    print("\n[4/5] Running LSTM model...")
    lstm_model.main()

    print("\n[5/5] Running statistical tests...")
    statistical_test.main()

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)