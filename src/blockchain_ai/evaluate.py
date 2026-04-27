import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(test_path: str, target_col: str, model_path: str, report_path: str) -> dict:
    df = pd.read_csv(test_path)
    X = df.drop(columns=[target_col])
    y_log = df[target_col]

    model = joblib.load(model_path)
    y_pred_log = model.predict(X)

    y_true = np.expm1(np.clip(y_log, a_min=None, a_max=709.0))
    y_pred = np.expm1(np.clip(y_pred_log, a_min=None, a_max=709.0))

    report = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
