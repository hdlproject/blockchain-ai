import pandas as pd
import pytest
import joblib
from blockchain_ai.config import TrainConfig
from blockchain_ai.train import train_model


def _processed_df():
    return pd.DataFrame({
        "value": [float(i * 1000) for i in range(20)],
        "gas": [21000] * 20,
        "max_fee_per_gas": [0.0] * 20,
        "max_priority_fee_per_gas": [0.0] * 20,
        "transaction_type": [0] * 10 + [2] * 10,
        "log_gas_price": [23.2] * 20,
    })


def _default_config(**overrides):
    base = dict(
        target_col="log_gas_price",
        model_type="xgboost",
        stratify_col="transaction_type",
        test_size=0.2,
        hyperparameters={"n_estimators": 10, "random_state": 42},
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_train_model_saves_model_artifact(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)

    train_model(str(csv_path), str(model_path), str(test_path), _default_config())

    assert model_path.exists()


def test_train_model_saves_test_split(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)

    train_model(str(csv_path), str(model_path), str(test_path), _default_config())

    assert test_path.exists()
    test_df = pd.read_csv(test_path)
    assert "log_gas_price" in test_df.columns
    assert len(test_df) == 4  # 20% of 20 rows


def test_train_model_raises_on_unknown_model_type(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(
            str(csv_path),
            str(tmp_path / "m.joblib"),
            str(tmp_path / "t.csv"),
            _default_config(model_type="unknown"),
        )


def test_train_model_uses_hyperparameters_from_config(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)

    train_model(
        str(csv_path), str(model_path), str(test_path),
        _default_config(hyperparameters={"n_estimators": 5, "random_state": 0}),
    )

    model = joblib.load(model_path)
    assert model.n_estimators == 5


def _processed_df_no_stratify():
    return pd.DataFrame({
        "base_fee_gwei": [10.0 + i * 0.1 for i in range(20)],
        "gas_used_ratio": [0.5] * 20,
        "base_fee_trend": [0.01] * 20,
        "hour_of_day": [i % 24 for i in range(20)],
        "day_of_week": [i % 7 for i in range(20)],
        "log_base_fee_gwei": [2.3 + i * 0.01 for i in range(20)],
    })


def test_train_model_works_without_stratify_col(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df_no_stratify().to_csv(csv_path, index=False)

    cfg = TrainConfig(
        target_col="log_base_fee_gwei",
        model_type="xgboost",
        stratify_col=None,
        test_size=0.2,
        hyperparameters={"n_estimators": 10, "random_state": 42},
    )
    train_model(str(csv_path), str(model_path), str(test_path), cfg)

    assert model_path.exists()


def _classification_df():
    return pd.DataFrame({
        "tx_count": [100.0, 50.0, 200.0, 10.0, 5.0, 300.0, 80.0, 20.0, 150.0, 60.0,
                     110.0, 55.0, 210.0, 15.0, 8.0, 310.0, 85.0, 25.0, 155.0, 65.0],
        "account_age_days": [365.0, 180.0, 730.0, 30.0, 10.0, 1000.0, 400.0, 60.0, 500.0, 200.0,
                             370.0, 185.0, 735.0, 35.0, 15.0, 1005.0, 405.0, 65.0, 505.0, 205.0],
        "label": (["sanctioned"] * 7 + ["scammer"] * 7 + ["phishing"] * 6),
    })


def _classification_config():
    return TrainConfig(
        target_col="label",
        model_type="xgboost",
        stratify_col=None,
        test_size=0.2,
        hyperparameters={"n_estimators": 10, "random_state": 42},
    )


def test_train_model_classification_saves_model(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    train_model(str(csv_path), str(model_path), str(test_path), _classification_config(), task="classification")
    assert model_path.exists()


def test_train_model_classification_test_split_has_encoded_labels(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    train_model(str(csv_path), str(model_path), str(test_path), _classification_config(), task="classification")
    test_df = pd.read_csv(test_path)
    assert "label" in test_df.columns
    assert set(test_df["label"].unique()).issubset({0, 1, 2})


def test_train_model_classification_model_has_predict_proba(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    import joblib as jl
    train_model(str(csv_path), str(model_path), str(tmp_path / "t.csv"), _classification_config(), task="classification")
    model = jl.load(model_path)
    assert hasattr(model, "predict_proba")
