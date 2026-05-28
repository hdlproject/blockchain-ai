import pytest
from blockchain_ai.config import load_config, PipelineConfig, IngestConfig, TrainConfig, HpoConfig, ServeConfig, FieldConfig, EtherscanConfig, CollectConfig


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


_VALID_YAML = """
ingest:
  feature_cols:
    - value
    - gas
  fill_zero_cols:
    - max_fee_per_gas
  target_col: gas_price

train:
  target_col: log_gas_price
  model_type: xgboost
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
    assert cfg.ingest.feature_cols == ["value", "gas"]
    assert cfg.ingest.fill_zero_cols == ["max_fee_per_gas"]
    assert cfg.ingest.target_col == "gas_price"


def test_load_config_train_fields(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.train.target_col == "log_gas_price"
    assert cfg.train.model_type == "xgboost"
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
  target_col: gas_price
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="train"):
        load_config(path)


def test_load_config_with_hpo_section(tmp_path):
    yaml_content = """
ingest:
  feature_cols: [value]
  fill_zero_cols: [max_fee_per_gas]
  target_col: gas_price
train:
  target_col: log_gas_price
  model_type: xgboost
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
  feature_cols: [value]
  fill_zero_cols: [max_fee_per_gas]
  target_col: gas_price
train:
  target_col: log_gas_price
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 10
hpo: {}
"""
    path = _write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="hpo.*n_trials"):
        load_config(path)


_SERVE_YAML = """
ingest:
  feature_cols:
    - value
    - gas
  fill_zero_cols: []
  target_col: gas_price

train:
  target_col: log_gas_price
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 10

serve:
  title: Test API
  description: A test predictor.
  model_path: models/gas_price_predictor.joblib
  target_description: Gas price
  target_unit: Wei
  log_transform: true
  fields:
    value:
      type: float
      description: ETH value in Wei.
      example: 0.0
      ge: 0
    gas:
      type: int
      description: Gas limit.
      example: 21000
      gt: 0
"""


def test_load_config_with_serve_section(tmp_path):
    path = _write_yaml(tmp_path, _SERVE_YAML)
    cfg = load_config(path)
    assert cfg.serve is not None
    assert isinstance(cfg.serve, ServeConfig)


def test_load_config_serve_fields(tmp_path):
    path = _write_yaml(tmp_path, _SERVE_YAML)
    cfg = load_config(path)
    s = cfg.serve
    assert s.title == "Test API"
    assert s.model_path == "models/gas_price_predictor.joblib"
    assert s.log_transform is True
    assert set(s.fields.keys()) == {"value", "gas"}


def test_load_config_serve_field_metadata(tmp_path):
    path = _write_yaml(tmp_path, _SERVE_YAML)
    cfg = load_config(path)
    value_field = cfg.serve.fields["value"]
    assert isinstance(value_field, FieldConfig)
    assert value_field.type == "float"
    assert value_field.ge == 0
    assert value_field.gt is None
    gas_field = cfg.serve.fields["gas"]
    assert gas_field.type == "int"
    assert gas_field.gt == 0


def test_load_config_serve_missing_required_key_raises(tmp_path):
    yaml_content = """
ingest:
  feature_cols: [value]
  fill_zero_cols: []
  target_col: gas_price
train:
  target_col: log_gas_price
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 10
serve:
  title: Broken
"""
    path = _write_yaml(tmp_path, yaml_content)
    with pytest.raises(ValueError, match="serve.*description"):
        load_config(path)


def test_load_config_without_serve_section(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.serve is None


_ETHERSCAN_YAML = """
etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 10

collect:
  n_blocks: 500
  output_path: data/raw/blocks.csv

ingest:
  feature_cols:
    - base_fee_gwei
    - gas_used_ratio
  fill_zero_cols: []
  target_col: base_fee_gwei

train:
  target_col: log_base_fee_gwei
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 10
    random_state: 42
"""


def test_load_config_with_etherscan_section(tmp_path):
    path = _write_yaml(tmp_path, _ETHERSCAN_YAML)
    cfg = load_config(path)
    assert cfg.etherscan is not None
    assert isinstance(cfg.etherscan, EtherscanConfig)
    assert cfg.etherscan.base_url == "https://api.etherscan.io/v2/api"
    assert cfg.etherscan.chain_id == 1
    assert cfg.etherscan.rate_limit_per_sec == 5
    assert cfg.etherscan.timeout_sec == 10


def test_load_config_with_collect_section(tmp_path):
    path = _write_yaml(tmp_path, _ETHERSCAN_YAML)
    cfg = load_config(path)
    assert cfg.collect is not None
    assert isinstance(cfg.collect, CollectConfig)
    assert cfg.collect.n_blocks == 500
    assert cfg.collect.output_path == "data/raw/blocks.csv"


    path = _write_yaml(tmp_path, _ETHERSCAN_YAML)
    cfg = load_config(path)


def test_load_config_without_etherscan_section(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.etherscan is None
    assert cfg.collect is None


_CLASSIFICATION_YAML = """
task: classification

goplus:
  base_url: https://api.gopluslabs.io/api/v1
  chain_id: 1
  rate_limit_per_sec: 2
  timeout_sec: 30

ofac:
  sdn_url: https://www.treasury.gov/ofac/downloads/sdn.csv
  timeout_sec: 60

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - tx_count
    - account_age_days
  fill_zero_cols: []
  target_col: label

train:
  target_col: label
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 10

serve:
  model_path: models/address_classifier.joblib
  confidence_threshold: 0.5
  db_path: data/jobs.db
"""


def test_load_classification_config(tmp_path):
    path = _write_yaml(tmp_path, _CLASSIFICATION_YAML)
    cfg = load_config(path)
    assert cfg.task == "classification"
    assert cfg.goplus is not None
    assert cfg.goplus.chain_id == 1
    assert cfg.ofac is not None
    assert cfg.ofac.timeout_sec == 60
    assert cfg.serve is not None
    assert cfg.serve.confidence_threshold == 0.5
    assert cfg.serve.db_path == "data/jobs.db"


def test_classification_config_missing_goplus_raises(tmp_path):
    yaml = _CLASSIFICATION_YAML.replace("goplus:\n  base_url: https://api.gopluslabs.io/api/v1\n  chain_id: 1\n  rate_limit_per_sec: 2\n  timeout_sec: 30\n\n", "")
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="goplus"):
        load_config(path)


def test_regression_config_defaults_task_to_regression(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.task == "regression"


def test_serve_classification_missing_db_path_raises(tmp_path):
    yaml = _CLASSIFICATION_YAML.replace("  db_path: data/jobs.db", "")
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="db_path"):
        load_config(path)


def test_load_clustering_config(tmp_path):
    yaml_content = """
task: clustering
etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30
collect:
  n_blocks: 100
  output_path: data/raw/transactions.csv
ingest:
  feature_cols: [value_eth, gas_price_gwei]
  fill_zero_cols: []
  target_col: ""
train:
  target_col: ""
  model_type: dbscan
  test_size: 0.0
  hyperparameters:
    eps: 0.5
    min_samples: 5
paths:
  processed_path: data/processed/transactions.csv
  report_path: reports/transaction_anomaly.json
serve:
  title: Test
  description: Test
  model_path: models/test.joblib
  fields:
    value_eth:
      type: float
      description: value
      example: 0.5
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)
    cfg = load_config(str(config_path))
    assert cfg.task == "clustering"
    assert cfg.train.model_type == "dbscan"
    assert cfg.paths.test_path == ""

def test_clustering_config_paths_test_path_optional(tmp_path):
    yaml_content = """
task: clustering
ingest:
  feature_cols: [value_eth]
  fill_zero_cols: []
  target_col: ""
train:
  target_col: ""
  model_type: dbscan
  test_size: 0.0
  hyperparameters:
    eps: 0.5
    min_samples: 5
paths:
  processed_path: data/processed/transactions.csv
  report_path: reports/transaction_anomaly.json
serve:
  title: Test
  description: Test
  model_path: models/test.joblib
  fields: {}
"""
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml_content)
    cfg = load_config(str(config_path))
    assert cfg.paths.test_path == ""
