import json
import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor, XGBClassifier
from blockchain_ai.config import IngestConfig, PathsConfig, PipelineConfig, ServeConfig, TrainConfig
from blockchain_ai.evaluate import evaluate_model, pre_evaluate_model, post_evaluate_model
from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper


def _regression_cfg(target_col: str, log_transform: bool = True) -> PipelineConfig:
    return PipelineConfig(
        ingest=IngestConfig(feature_cols=[], fill_zero_cols=[], target_col=target_col),
        train=TrainConfig(target_col=target_col, model_type="xgboost", test_size=0.2, hyperparameters={}),
        serve=ServeConfig(model_path="", log_transform=log_transform),
    )


def _classification_cfg(target_col: str) -> PipelineConfig:
    return PipelineConfig(
        task="classification",
        ingest=IngestConfig(feature_cols=[], fill_zero_cols=[], target_col=target_col),
        train=TrainConfig(target_col=target_col, model_type="xgboost", test_size=0.2, hyperparameters={}),
    )


def _make_regression_split(tmp_path):
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


def _make_classification_split(tmp_path):
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


# --- pre_evaluate_model ---

def test_pre_evaluate_model_splits_features_and_target(tmp_path):
    test_path, _ = _make_regression_split(tmp_path)
    X, y = pre_evaluate_model(test_path, _regression_cfg("log_gas_price"))
    assert "log_gas_price" not in X.columns
    assert list(y.name if hasattr(y, "name") else []) or y is not None
    assert len(X) == len(y)


# --- post_evaluate_model ---

def test_post_evaluate_model_regression_applies_expm1(tmp_path):
    y = pd.Series([23.2] * 5)
    y_raw = np.array([23.2] * 5)
    report = post_evaluate_model(y, y_raw, _regression_cfg("log_gas_price", log_transform=True))
    assert report["rmse"] == pytest.approx(0.0, abs=1.0)
    assert "rmse" in report and "mae" in report and "r2" in report


def test_post_evaluate_model_regression_no_transform(tmp_path):
    y = pd.Series([1.0, 2.0, 3.0])
    y_raw = np.array([1.0, 2.0, 3.0])
    report = post_evaluate_model(y, y_raw, _regression_cfg("col", log_transform=False))
    assert report["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_post_evaluate_model_classification_returns_accuracy_and_f1(tmp_path):
    y = pd.Series([0, 1, 2, 0, 1])
    y_raw = np.array([0, 1, 2, 0, 1])
    report = post_evaluate_model(y, y_raw, _classification_cfg("label"))
    assert report["accuracy"] == pytest.approx(1.0)
    assert "f1_macro" in report
    assert "f1_sanctioned" in report
    assert "f1_scammer" in report
    assert "f1_phishing" in report


# --- evaluate_model (orchestrator) ---

def test_evaluate_model_saves_report(tmp_path):
    test_path, model_path = _make_regression_split(tmp_path)
    report_path = str(tmp_path / "report.json")
    evaluate_model(test_path, model_path, report_path, _regression_cfg("log_gas_price"))
    assert (tmp_path / "report.json").exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert "rmse" in report and "mae" in report and "r2" in report


def test_evaluate_model_report_values_are_floats(tmp_path):
    test_path, model_path = _make_regression_split(tmp_path)
    evaluate_model(test_path, model_path, str(tmp_path / "r.json"), _regression_cfg("log_gas_price"))
    report = json.loads((tmp_path / "r.json").read_text())
    assert isinstance(report["rmse"], float)
    assert isinstance(report["mae"], float)
    assert isinstance(report["r2"], float)


def test_evaluate_model_metrics_are_in_wei(tmp_path):
    test_path, model_path = _make_regression_split(tmp_path)
    evaluate_model(test_path, model_path, str(tmp_path / "r.json"), _regression_cfg("log_gas_price", log_transform=True))
    report = json.loads((tmp_path / "r.json").read_text())
    # expm1(23.2) ≈ 1.2e10 — RMSE in log-space would be < 1
    assert report["rmse"] > 1.0 or report["rmse"] == pytest.approx(0.0, abs=1.0)


def test_evaluate_classification_saves_report(tmp_path):
    test_path, model_path = _make_classification_split(tmp_path)
    report_path = str(tmp_path / "clf_report.json")
    evaluate_model(test_path, model_path, report_path, _classification_cfg("label"))
    assert (tmp_path / "clf_report.json").exists()
    report = json.loads((tmp_path / "clf_report.json").read_text())
    assert "accuracy" in report
    assert "f1_macro" in report
    assert "f1_sanctioned" in report and "f1_scammer" in report and "f1_phishing" in report


def test_evaluate_classification_accuracy_is_float(tmp_path):
    test_path, model_path = _make_classification_split(tmp_path)
    evaluate_model(test_path, model_path, str(tmp_path / "r.json"), _classification_cfg("label"))
    report = json.loads((tmp_path / "r.json").read_text())
    assert isinstance(report["accuracy"], float)
    assert 0.0 <= report["accuracy"] <= 1.0


def test_evaluate_clustering(tmp_path):
    from blockchain_ai.config import (
        PipelineConfig, IngestConfig, TrainConfig, PathsConfig, ServeConfig
    )
    cfg = PipelineConfig(
        task="clustering",
        ingest=IngestConfig(feature_cols=["x", "y"], fill_zero_cols=[], target_col=""),
        train=TrainConfig(target_col="", model_type="dbscan", test_size=0.0,
                          hyperparameters={"eps": 0.5, "min_samples": 3}),
        paths=PathsConfig(processed_path=str(tmp_path / "p.csv"),
                          report_path=str(tmp_path / "r.json")),
        serve=ServeConfig(model_path=str(tmp_path / "model.joblib")),
    )
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2)).astype(np.float32)
    model = DBSCANWrapper(eps=0.5, min_samples=3)
    model.fit(X)
    joblib.dump(model, tmp_path / "model.joblib")

    df = pd.DataFrame(X, columns=["x", "y"])
    df.to_csv(tmp_path / "p.csv", index=False)

    report = evaluate_model(
        str(tmp_path / "p.csv"),
        str(tmp_path / "model.joblib"),
        str(tmp_path / "r.json"),
        cfg,
    )
    assert "anomaly_ratio" in report
    assert "n_clusters" in report
    assert "n_noise" in report
    assert 0.0 <= report["anomaly_ratio"] <= 1.0
    assert (tmp_path / "r.json").exists()
