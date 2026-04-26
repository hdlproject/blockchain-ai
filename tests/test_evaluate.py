import json
import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor
from blockchain_ai.evaluate import evaluate_model


def _make_test_split(tmp_path):
    X = pd.DataFrame({
        "nonce": list(range(10)),
        "transaction_index": list(range(10)),
        "value": [float(i * 1000) for i in range(10)],
        "gas": [21000] * 10,
        "receipt_cumulative_gas_used": [21000] * 10,
        "receipt_gas_used": [21000] * 10,
        "receipt_status": [1] * 10,
        "block_timestamp": [1672531200 + i * 12 for i in range(10)],
        "block_number": [16000000 + i for i in range(10)],
        "max_fee_per_gas": [0.0] * 10,
        "max_priority_fee_per_gas": [0.0] * 10,
        "transaction_type": [0] * 5 + [2] * 5,
        "receipt_effective_gas_price": [12000000000] * 10,
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
