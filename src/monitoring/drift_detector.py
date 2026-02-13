import json
import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(training_stats_path, production_data_df):
    training_stats = pd.read_json(training_stats_path)

    drift_results = {}

    for col in production_data_df.columns:
        train_mean = training_stats[col]["mean"]
        prod_mean = production_data_df[col].mean()

        drift_results[col] = abs(train_mean - prod_mean)

    return drift_results


