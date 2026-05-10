import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from blockchain_ai.config import PipelineConfig
from blockchain_ai.predict import LABEL_ENCODER


def post_process_predictions(
    y: pd.Series,
    y_raw: np.ndarray,
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if log_transform:
        y_true = np.expm1(np.clip(np.asarray(y, dtype=np.float64), 0.0, 709.0))
        y_pred = np.expm1(np.clip(np.asarray(y_raw, dtype=np.float64), 0.0, 709.0))
    else:
        y_true = np.asarray(y)
        y_pred = np.asarray(y_raw)
    return y_true, y_pred


def evaluate_model(
    test_path: str,
    model_path: str,
    report_path: str,
    cfg: PipelineConfig,
) -> dict:
    target_col = cfg.train.target_col
    log_transform = cfg.serve.log_transform if cfg.serve else False

    df = pd.read_csv(test_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = joblib.load(model_path)
    y_raw = model.predict(X)
    y_true, y_pred = post_process_predictions(y, y_raw, log_transform)

    if cfg.task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        per_class = {
            f"f1_{name}": float(f1_score(y_true, y_pred, average=None, labels=[idx], zero_division=0)[0])
            for idx, name in LABEL_ENCODER.items()
        }
        report = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            **per_class,
        }
    else:
        report = {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
