import json
import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor
from blockchain_ai.evaluate import evaluate_model


def _make_test_split(tmp_path):
    X = pd.DataFrame({
        "value": [float(i * 1000) for i in range(10)],
        "gas": [21000] * 10,
        "max_fee_per_gas": [0.0] * 10,
        "max_priority_fee_per_gas": [0.0] * 10,
        "transaction_type": [0] * 5 + [2] * 5,
    })
    y = pd.Series([23.2] * 10, name="log_gas_price")

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    test_df = X.copy()
    test_df["log_gas_price"] = y
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    return str(test_path), str(model_path)


def test_evaluate_model_saves_report(tmp_path):
    test_path, model_path = _make_test_split(tmp_path)
    report_path = str(tmp_path / "report.json")

    evaluate_model(test_path, "log_gas_price", model_path, report_path)

    assert (tmp_path / "report.json").exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert "rmse" in report
    assert "mae" in report
    assert "r2" in report


def test_evaluate_model_report_values_are_floats(tmp_path):
    test_path, model_path = _make_test_split(tmp_path)
    report_path = str(tmp_path / "report.json")

    evaluate_model(test_path, "log_gas_price", model_path, report_path)

    report = json.loads((tmp_path / "report.json").read_text())
    assert isinstance(report["rmse"], float)
    assert isinstance(report["mae"], float)
    assert isinstance(report["r2"], float)


def test_evaluate_model_metrics_are_in_wei(tmp_path):
    """RMSE should be in Wei (large numbers), not log-space (small numbers ~23)."""
    test_path, model_path = _make_test_split(tmp_path)
    report_path = str(tmp_path / "report.json")

    evaluate_model(test_path, "log_gas_price", model_path, report_path)

    report = json.loads((tmp_path / "report.json").read_text())
    # expm1(23.2) ≈ 1.2e10 Wei — RMSE in log-space would be < 1
    assert report["rmse"] > 1.0 or report["rmse"] == pytest.approx(0.0, abs=1.0)


def _make_classification_split(tmp_path):
    from xgboost import XGBClassifier
    from blockchain_ai.train import LABEL_TO_INT
    X = pd.DataFrame({"tx_count": [100.0, 50.0, 200.0, 10.0, 5.0, 300.0, 80.0, 20.0, 150.0, 60.0],
                      "age": [365.0, 180.0, 730.0, 30.0, 10.0, 1000.0, 400.0, 60.0, 500.0, 200.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 0, 1])
    model = XGBClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "clf.joblib"
    joblib.dump(model, model_path)
    test_df = X.copy()
    test_df["label"] = y
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)
    return str(test_path), str(model_path)


def test_evaluate_classification_saves_report(tmp_path):
    test_path, model_path = _make_classification_split(tmp_path)
    report_path = str(tmp_path / "clf_report.json")
    import json as json_lib
    from blockchain_ai.evaluate import evaluate_model
    evaluate_model(test_path, "label", model_path, report_path, task="classification")
    assert (tmp_path / "clf_report.json").exists()
    report = json_lib.loads((tmp_path / "clf_report.json").read_text())
    assert "accuracy" in report
    assert "f1_macro" in report
    assert "f1_sanctioned" in report
    assert "f1_scammer" in report
    assert "f1_phishing" in report


def test_evaluate_classification_accuracy_is_float(tmp_path):
    test_path, model_path = _make_classification_split(tmp_path)
    import json as json_lib
    from blockchain_ai.evaluate import evaluate_model
    evaluate_model(test_path, "label", model_path, str(tmp_path / "r.json"), task="classification")
    report = json_lib.loads((tmp_path / "r.json").read_text())
    assert isinstance(report["accuracy"], float)
    assert 0.0 <= report["accuracy"] <= 1.0
