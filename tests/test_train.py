import pandas as pd
import pytest
import joblib
from blockchain_ai.config import TrainConfig
from blockchain_ai.train import train_model


def _processed_df():
    return pd.DataFrame({
        "nonce": list(range(20)),
        "transaction_index": list(range(20)),
        "value": [float(i * 1000) for i in range(20)],
        "gas": [21000] * 20,
        "receipt_cumulative_gas_used": [21000] * 20,
        "receipt_gas_used": [21000] * 20,
        "receipt_status": [1] * 20,
        "block_timestamp": [1672531200 + i * 12 for i in range(20)],
        "block_number": [16000000 + i for i in range(20)],
        "max_fee_per_gas": [0.0] * 20,
        "max_priority_fee_per_gas": [0.0] * 20,
        "transaction_type": [0] * 10 + [2] * 10,
        "receipt_effective_gas_price": [12000000000] * 20,
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
