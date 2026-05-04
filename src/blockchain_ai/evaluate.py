import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(
    test_path: str,
    target_col: str,
    model_path: str,
    report_path: str,
    task: str = "regression",
) -> dict:
    df = pd.read_csv(test_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = joblib.load(model_path)

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(X)
        report = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "f1_sanctioned": float(f1_score(y, y_pred, average=None, labels=[0], zero_division=0)[0]),
            "f1_scammer": float(f1_score(y, y_pred, average=None, labels=[1], zero_division=0)[0]),
            "f1_phishing": float(f1_score(y, y_pred, average=None, labels=[2], zero_division=0)[0]),
        }
    else:
        y_log = y
        y_pred_log = model.predict(X)
        y_true = np.expm1(np.clip(np.asarray(y_log, dtype=np.float64), 0.0, 709.0))
        y_pred = np.expm1(np.clip(np.asarray(y_pred_log, dtype=np.float64), 0.0, 709.0))
        report = {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
