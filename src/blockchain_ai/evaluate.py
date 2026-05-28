import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score

from blockchain_ai.config import PipelineConfig
from blockchain_ai.predict import LABEL_ENCODER


def pre_evaluate_model(test_path: str, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    target_col = cfg.train.target_col
    df = pd.read_csv(test_path)
    return df.drop(columns=[target_col]), df[target_col]


def post_evaluate_model(
        y: pd.Series,
        y_raw: np.ndarray,
        cfg: PipelineConfig,
) -> dict:
    log_transform = cfg.serve.log_transform if cfg.serve else False

    if log_transform:
        y_true = np.expm1(np.clip(np.asarray(y, dtype=np.float64), 0.0, 709.0))
        y_pred = np.expm1(np.clip(np.asarray(y_raw, dtype=np.float64), 0.0, 709.0))
    else:
        y_true = np.asarray(y)
        y_pred = np.asarray(y_raw)

    if cfg.task == "classification":
        per_class = {
            f"f1_{name}": float(f1_score(y_true, y_pred, average=None, labels=[idx], zero_division=0)[0])
            for idx, name in LABEL_ENCODER.items()
        }
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            **per_class,
        }
    elif cfg.task == "regression":
        return {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    else:
        raise ValueError(f"Unsupported task type: {cfg.task}")


def evaluate_model(
        test_path: str,
        model_path: str,
        report_path: str,
        cfg: PipelineConfig,
) -> dict:
    if cfg.task == "clustering":
        df = pd.read_csv(test_path)
        X = df[cfg.ingest.feature_cols].values
        model = joblib.load(model_path)
        labels = model.predict(X)
        n_noise = int(np.sum(labels == -1))
        n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        report = {
            "anomaly_ratio": round(float(n_noise / len(labels)), 4),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
        return report

    X, y = pre_evaluate_model(test_path, cfg)
    y_raw = joblib.load(model_path).predict(X)
    report = post_evaluate_model(y, y_raw, cfg)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
