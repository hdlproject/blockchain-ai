import numpy as np
import pandas as pd
import pytest
import joblib
from blockchain_ai.config import IngestConfig, PathsConfig, PipelineConfig, ServeConfig, TrainConfig
from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper
from blockchain_ai.train import train_model


def _cfg(target_col: str, model_type: str = "xgboost", hyperparameters: dict | None = None,
         task: str = "regression") -> PipelineConfig:
    return PipelineConfig(
        task=task,
        ingest=IngestConfig(feature_cols=[], fill_zero_cols=[], target_col=target_col),
        train=TrainConfig(
            target_col=target_col,
            model_type=model_type,
            test_size=0.2,
            hyperparameters=hyperparameters or {"n_estimators": 10, "random_state": 42},
        ),
    )


def _processed_df():
    return pd.DataFrame({
        "value": [float(i * 1000) for i in range(20)],
        "gas": [21000] * 20,
        "max_fee_per_gas": [0.0] * 20,
        "max_priority_fee_per_gas": [0.0] * 20,
        "transaction_type": [0] * 10 + [2] * 10,
        "log_gas_price": [23.2] * 20,
    })


def _classification_df():
    return pd.DataFrame({
        "tx_count": [100.0, 50.0, 200.0, 10.0, 5.0, 300.0, 80.0, 20.0, 150.0, 60.0,
                     110.0, 55.0, 210.0, 15.0, 8.0, 310.0, 85.0, 25.0, 155.0, 65.0],
        "account_age_days": [365.0, 180.0, 730.0, 30.0, 10.0, 1000.0, 400.0, 60.0, 500.0, 200.0,
                             370.0, 185.0, 735.0, 35.0, 15.0, 1005.0, 405.0, 65.0, 505.0, 205.0],
        "label": (["sanctioned"] * 7 + ["scammer"] * 7 + ["phishing"] * 6),
    })


def test_train_model_saves_model_artifact(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)
    train_model(str(csv_path), str(tmp_path / "m.joblib"), str(tmp_path / "t.csv"),
                _cfg("log_gas_price"))
    assert (tmp_path / "m.joblib").exists()


def test_train_model_saves_test_split(tmp_path):
    csv_path = tmp_path / "processed.csv"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)
    train_model(str(csv_path), str(tmp_path / "m.joblib"), str(test_path), _cfg("log_gas_price"))
    assert test_path.exists()
    test_df = pd.read_csv(test_path)
    assert "log_gas_price" in test_df.columns
    assert len(test_df) == 4  # 20% of 20 rows


def test_train_model_raises_on_unknown_model_type(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(str(csv_path), str(tmp_path / "m.joblib"), str(tmp_path / "t.csv"),
                    _cfg("log_gas_price", model_type="unknown"))


def test_train_model_uses_hyperparameters_from_config(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    _processed_df().to_csv(csv_path, index=False)
    train_model(str(csv_path), str(model_path), str(tmp_path / "t.csv"),
                _cfg("log_gas_price", hyperparameters={"n_estimators": 5, "random_state": 0}))
    assert joblib.load(model_path).n_estimators == 5


def test_train_model_classification_saves_model(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    train_model(str(csv_path), str(tmp_path / "m.joblib"), str(tmp_path / "t.csv"),
                _cfg("label", task="classification"))
    assert (tmp_path / "m.joblib").exists()


def test_train_model_classification_test_split_has_encoded_labels(tmp_path):
    csv_path = tmp_path / "features.csv"
    test_path = tmp_path / "test.csv"
    _classification_df().to_csv(csv_path, index=False)
    train_model(str(csv_path), str(tmp_path / "m.joblib"), str(test_path),
                _cfg("label", task="classification"))
    test_df = pd.read_csv(test_path)
    assert "label" in test_df.columns
    assert set(test_df["label"].unique()).issubset({0, 1, 2})


def test_train_model_classification_model_has_predict_proba(tmp_path):
    csv_path = tmp_path / "features.csv"
    model_path = tmp_path / "model.joblib"
    _classification_df().to_csv(csv_path, index=False)
    train_model(str(csv_path), str(model_path), str(tmp_path / "t.csv"),
                _cfg("label", task="classification"))
    assert hasattr(joblib.load(model_path), "predict_proba")


def test_train_model_lstm_saves_model_and_has_predict(tmp_path):
    n = 50
    feature_cols = ["base_fee_gwei", "gas_used_ratio", "hour_of_day", "day_of_week", "base_fee_trend"]
    data = pd.DataFrame({
        "base_fee_gwei": [15.0 + i * 0.01 for i in range(n)],
        "gas_used_ratio": [0.5] * n,
        "hour_of_day": [i % 24 for i in range(n)],
        "day_of_week": [i % 7 for i in range(n)],
        "base_fee_trend": [0.01] * n,
        "log_next_base_fee_gwei": [2.7 + i * 0.001 for i in range(n)],
    })
    csv_path = tmp_path / "blocks.csv"
    model_path = tmp_path / "model.joblib"
    data.to_csv(csv_path, index=False)
    train_model(
        str(csv_path),
        str(model_path),
        str(tmp_path / "test.csv"),
        _cfg(
            "log_next_base_fee_gwei",
            model_type="lstm",
            hyperparameters={
                "sequence_length": 5,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "epochs": 2,
                "lr": 0.01,
                "batch_size": 4,
            },
        ),
    )
    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "sequence_length")


def test_train_clustering_returns_dbscan_wrapper(tmp_path):
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
    df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    df.to_csv(tmp_path / "p.csv", index=False)

    model = train_model(
        str(tmp_path / "p.csv"),
        str(tmp_path / "model.joblib"),
        "",
        cfg,
    )
    assert isinstance(model, DBSCANWrapper)
    assert (tmp_path / "model.joblib").exists()
