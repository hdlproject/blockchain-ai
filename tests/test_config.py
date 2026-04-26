import pytest
from blockchain_ai.config import load_config, PipelineConfig, IngestConfig, TrainConfig, HpoConfig


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


_VALID_YAML = """
ingest:
  feature_cols:
    - nonce
    - value
  fill_zero_cols:
    - max_fee_per_gas
  timestamp_col: block_timestamp
  target_col: gas_price

train:
  target_col: log_gas_price
  model_type: xgboost
  stratify_col: transaction_type
  test_size: 0.2
  hyperparameters:
    n_estimators: 10
    random_state: 42
"""


def test_load_config_returns_pipeline_config(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert isinstance(cfg, PipelineConfig)
    assert isinstance(cfg.ingest, IngestConfig)
    assert isinstance(cfg.train, TrainConfig)


def test_load_config_ingest_fields(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.ingest.feature_cols == ["nonce", "value"]
    assert cfg.ingest.fill_zero_cols == ["max_fee_per_gas"]
    assert cfg.ingest.timestamp_col == "block_timestamp"
    assert cfg.ingest.target_col == "gas_price"


def test_load_config_train_fields(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.train.target_col == "log_gas_price"
    assert cfg.train.model_type == "xgboost"
    assert cfg.train.stratify_col == "transaction_type"
    assert cfg.train.test_size == 0.2
    assert cfg.train.hyperparameters == {"n_estimators": 10, "random_state": 42}


def test_load_config_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))


def test_load_config_raises_on_missing_ingest_key(tmp_path):
    yaml = """
train:
  target_col: log_gas_price
  model_type: xgboost
  stratify_col: transaction_type
  test_size: 0.2
  hyperparameters: {}
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="ingest"):
        load_config(path)


def test_load_config_raises_on_missing_train_key(tmp_path):
    yaml = """
ingest:
  feature_cols: []
  fill_zero_cols: []
  timestamp_col: block_timestamp
  target_col: gas_price
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="train"):
        load_config(path)


def test_load_config_with_hpo_section(tmp_path):
    yaml_content = """
ingest:
  feature_cols: [nonce]
  fill_zero_cols: [max_fee_per_gas]
  timestamp_col: block_timestamp
  target_col: gas_price
train:
  target_col: log_gas_price
  model_type: xgboost
  stratify_col: transaction_type
  test_size: 0.2
  hyperparameters:
    n_estimators: 10
hpo:
  n_trials: 20
"""
    path = _write_yaml(tmp_path, yaml_content)
    cfg = load_config(path)
    assert cfg.hpo is not None
    assert isinstance(cfg.hpo, HpoConfig)
    assert cfg.hpo.n_trials == 20


def test_load_config_without_hpo_section(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.hpo is None


def test_hpo_config_missing_n_trials_raises(tmp_path):
    yaml_content = """
ingest:
  feature_cols: [nonce]
  fill_zero_cols: [max_fee_per_gas]
  timestamp_col: block_timestamp
  target_col: gas_price
train:
  target_col: log_gas_price
  model_type: xgboost
  stratify_col: transaction_type
  test_size: 0.2
  hyperparameters:
    n_estimators: 10
hpo: {}
"""
    path = _write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="hpo.*n_trials"):
        load_config(path)
