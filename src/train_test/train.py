import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==========================================================
# CONFIG
# ==========================================================

@dataclass
class Config:
    data_path: Path
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.1765
    n_trials: int = 50
    cv_folds: int = 5

    feature_columns: list = field(default_factory=lambda: [
        "ipv", "ips", "iaa", "ieg", "no_av", "ida", "media"
    ])
    target_column: str = "defasagem"

    def __post_init__(self):
        self.models_dir = Path("src/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)


# ==========================================================
# DATA
# ==========================================================

def load_data(config: Config) -> pd.DataFrame:
    df = pd.read_csv(config.data_path)
    if "nº_av" in df.columns:
        df = df.rename(columns={"nº_av": "no_av"})

    required = config.feature_columns + [config.target_column]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def split_data(config: Config, df: pd.DataFrame):
    X = df[config.feature_columns]
    y = df[config.target_column]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.val_size, stratify=y_temp, random_state=config.random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================================
# TRAINING STATS
# ==========================================================

def save_train_stats(config: Config, X_train: pd.DataFrame):
    """
    Salva estatísticas do treino para monitoramento de data drift.
    """
    train_stats = {}
    for col in X_train.columns:
        train_stats[col] = {
            "mean": X_train[col].mean(),
            "std": X_train[col].std(),
            "sample": X_train[col].sample(n=min(100, len(X_train)), random_state=config.random_state).tolist()
        }


    stats_path = config.models_dir / f"train_stats.json"
    with open(stats_path, "w") as f:
        json.dump(train_stats, f, indent=2)

    logger.info(f"Training stats saved at: {stats_path}")
    return stats_path


# ==========================================================
# OPTUNA
# ==========================================================

def optimize(config: Config, X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": config.random_state,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "verbosity": 0
        }
        model = XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        scores = cross_validate(model, X_train, y_train, cv=skf, scoring="recall", n_jobs=-1)
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=config.random_state))
    study.optimize(objective, n_trials=config.n_trials)

    best_params = study.best_params
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "random_state": config.random_state,
        "eval_metric": "logloss",
        "n_jobs": -1
    })
    return best_params


# ==========================================================
# TRAIN + EVAL
# ==========================================================

def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "recall": recall_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


# ==========================================================
# SAVE
# ==========================================================

def save_artifacts(config: Config, model, params, metrics):

    model_path = config.models_dir / f"model.pkl"
    joblib.dump(model, model_path)

    with open(config.models_dir / f"params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(config.models_dir / f"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Model and artifacts saved in: {config.models_dir}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("data/processed/students_feature_engineering.csv"))
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    config = Config(data_path=args.data_path, n_trials=args.n_trials)
    df = load_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(config, df)

    # salvar estatísticas do treino para drift
    train_stats_path = save_train_stats(config, X_train)

    logger.info("Running Optuna...")
    best_params = optimize(config, X_train, y_train)

    logger.info("Training final model...")
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    logger.info("Evaluating...")
    test_metrics = evaluate(model, X_test, y_test)

    save_artifacts(config, model, best_params, test_metrics)
    logger.info("Training completed successfully.")
    logger.info(f"Use this path for drift detection: {train_stats_path}")


if __name__ == "__main__":
    main()
