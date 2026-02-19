# test_monitoring.py
import pandas as pd
from src.monitoring.logger import log_inference
from src.monitoring.prediction_drift import update_prediction_metrics, load_predictions
from src.monitoring.performance_tracker import evaluate_performance
from src.monitoring.model_health import health_report
from src.monitoring.drift_detector import detect_and_set_gauges



# ---------------------------------------------------
# 1️⃣ Criar log fake de inferência
# ---------------------------------------------------
print("\n=== Criando log de teste ===")
test_input = {"ipv": 0.5, "ips": 0.6, "iaa": 0.4, "ieg": 0.7, "no_av": 0.3, "ida": 0.5}
test_prediction = 0.8
test_latency = 0.02
log_inference(test_input, test_prediction, test_latency, true_label=1)

# ---------------------------------------------------
# 2️⃣ Testar Prediction Drift
# ---------------------------------------------------
print("\n=== Testando Prediction Drift ===")
df_preds = load_predictions()
print("Últimas predições carregadas:")
print(df_preds.tail())
update_prediction_metrics()

# ---------------------------------------------------
# 3️⃣ Testar Performance Tracker
# ---------------------------------------------------
print("\n=== Testando Performance Tracker ===")
evaluate_performance()

# ---------------------------------------------------
# 4️⃣ Testar Model Health
# ---------------------------------------------------
print("\n=== Testando Model Health ===")
health_report()

# ---------------------------------------------------
# 5️⃣ Testar Data Drift
# ---------------------------------------------------
print("\n=== Testando Data Drift ===")
df_test = pd.DataFrame([test_input])
df_test["media"] = df_test[['ipv','ips','iaa','ieg','no_av','ida']].mean(axis=1)
drift_results = detect_and_set_gauges("src/models/train_stats.json", df_test)
print("Drift Results:", drift_results)

print("\n✅ Todos os módulos de monitoramento testados com sucesso!")
