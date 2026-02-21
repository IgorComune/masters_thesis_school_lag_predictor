# src/monitoring/drift_detector.py
import pandas as pd
from scipy.stats import ks_2samp
from . import metrics

def detect_and_set_gauges(training_stats_path, production_data_df):
    # supondo training_stats_path seja um json com estatísticas simples por coluna
    training_stats = pd.read_json(training_stats_path)
    for col in production_data_df.columns:
        # obter arrays reais: se training_stats guarda distribuições, use-as; aqui, simplificação:
        # Vamos calcular KS entre training sample e prod sample (você precisa do sample de treinamento)
        # se você tiver apenas estatísticas, adapte para outro teste.
        # Exemplo mínimo:
        try:
            train_sample = pd.Series(training_stats[col]["sample"])  # se existir
            prod_sample = production_data_df[col].dropna()
            ks_stat, pval = ks_2samp(train_sample, prod_sample)
            metrics.set_drift_for_feature(col, ks_stat)
        except Exception:
            # fallback: comparar médias (menos confiável)
            ks_approx = abs(training_stats[col]["mean"] - production_data_df[col].mean())
            metrics.set_drift_for_feature(col, ks_approx)
