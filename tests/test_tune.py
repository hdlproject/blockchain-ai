import pandas as pd
import pytest
from blockchain_ai.config import TrainConfig
from blockchain_ai.tune import run_hpo


def _processed_df():
    return pd.DataFrame({
        "value": [float(i * 1000) for i in range(40)],
        "gas": [21000] * 40,
        "max_fee_per_gas": [0.0] * 40,
        "max_priority_fee_per_gas": [0.0] * 40,
        "transaction_type": [0] * 20 + [2] * 20,
        "log_gas_price": [23.2 + i * 0.01 for i in range(40)],
    })


def _base_config(**overrides):
    base = dict(
        target_col="log_gas_price",
        model_type="xgboost",
        stratify_col="transaction_type",
        test_size=0.2,
        hyperparameters={"n_estimators": 5, "random_state": 42},
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_run_hpo_returns_train_config(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)

    result = run_hpo(str(csv_path), _base_config(), n_trials=3)

    assert isinstance(result, TrainConfig)


def test_run_hpo_replaces_hyperparameters(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)

    original_config = _base_config(hyperparameters={"n_estimators": 5, "random_state": 42})
    result = run_hpo(str(csv_path), original_config, n_trials=3)

    assert result.hyperparameters != original_config.hyperparameters


def test_run_hpo_preserves_non_hyperparameter_fields(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)

    original_config = _base_config()
    result = run_hpo(str(csv_path), original_config, n_trials=3)

    assert result.target_col == original_config.target_col
    assert result.model_type == original_config.model_type
    assert result.stratify_col == original_config.stratify_col
    assert result.test_size == original_config.test_size


def test_run_hpo_result_has_required_xgboost_keys(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)

    result = run_hpo(str(csv_path), _base_config(), n_trials=3)

    for key in ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"):
        assert key in result.hyperparameters, f"Missing key: {key}"
