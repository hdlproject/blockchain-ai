import json
import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from blockchain_ai.evaluate import evaluate_model


def test_evaluate_model_saves_report(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    df.to_csv(csv_path, index=False)

    model = LinearRegression()
    model.fit(df[["feature1"]], df["target"])
    joblib.dump(model, model_path)

    evaluate_model(str(csv_path), "target", str(model_path), str(report_path))

    report = json.loads(report_path.read_text())
    assert "rmse" in report
    assert "mae" in report
    assert "r2" in report
    assert report["r2"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_model_report_values_are_floats(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "target": [1.5, 2.5, 3.5]})
    df.to_csv(csv_path, index=False)
    model = LinearRegression()
    model.fit(df[["feature1"]], df["target"])
    joblib.dump(model, model_path)

    evaluate_model(str(csv_path), "target", str(model_path), str(report_path))

    report = json.loads(report_path.read_text())
    assert isinstance(report["rmse"], float)
    assert isinstance(report["mae"], float)
    assert isinstance(report["r2"], float)
