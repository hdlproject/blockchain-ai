import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(input_path: str, target_col: str, model_path: str, report_path: str) -> dict:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = joblib.load(model_path)
    predictions = model.predict(X)

    report = {
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
