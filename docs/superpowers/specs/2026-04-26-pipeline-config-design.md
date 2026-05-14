# Pipeline Config File — Design Spec

**Date:** 2026-04-26

---

## Goal

Replace hardcoded constants in `ingest.py` and `train.py` with a YAML config file per experiment. Adding a new model on new data requires only a new YAML — no code changes.

---

## 1. Config File Structure

Location: `configs/<experiment-name>.yaml`

```yaml
# configs/ethereum-gas-price-predictor.yaml

ingest:
  drop_cols:
    - hash
    - block_hash
    - from_address
    - to_address
    - input
    - receipt_contract_address
    - receipt_root
  fill_zero_cols:
    - max_fee_per_gas
    - max_priority_fee_per_gas
  timestamp_col: block_timestamp
  target_col: gas_price

train:
  target_col: log_gas_price
  model_type: xgboost
  stratify_col: transaction_type
  test_size: 0.2
  hyperparameters:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
```

A new experiment = a new YAML file. No code changes required.

---

## 2. Config Loading (`src/blockchain_ai/config.py`)

New module. Loads and validates YAML into typed dataclasses.

```python
@dataclass
class IngestConfig:
    drop_cols: list[str]
    fill_zero_cols: list[str]
    timestamp_col: str
    target_col: str

@dataclass
class TrainConfig:
    target_col: str
    model_type: str
    stratify_col: str
    test_size: float
    hyperparameters: dict

@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig

def load_config(path: str) -> PipelineConfig: ...
```

Validation: raises `ValueError` for any missing required key. No external schema library.

---

## 3. Pipeline Changes

### `ingest.py`

- Remove `_DROP_COLS` and `_EIP1559_COLS` module-level constants
- Signature: `load_and_clean(input_path, output_path, config: IngestConfig)`
- Uses `config.drop_cols`, `config.fill_zero_cols`, `config.timestamp_col`, `config.target_col`

### `train.py`

- Remove `model_type` and `target_col` as separate parameters
- Signature: `train_model(input_path, model_path, test_path, config: TrainConfig)`
- Uses `config.target_col`, `config.model_type`, `config.stratify_col`, `config.test_size`, `config.hyperparameters`
- XGBRegressor instantiated with `**config.hyperparameters`

### `evaluate.py` and `predict.py`

Unchanged — no hardcoded config.

### `run_pipeline.py`

- Replace `--target` and `--model-type` flags with single `--config` flag
- Loads `PipelineConfig` at startup, passes `config.ingest` and `config.train` to respective functions

```bash
poetry run python scripts/run_pipeline.py \
    --raw data/raw/ethereum-transactions.zip \
    --config configs/ethereum-gas-price-predictor.yaml
```

---

## 4. Dependencies

Add `pyyaml>=6.0` to `pyproject.toml`.

---

## File Map

| File | Action |
|---|---|
| `configs/ethereum-gas-price-predictor.yaml` | Create |
| `src/blockchain_ai/config.py` | Create |
| `src/blockchain_ai/ingest.py` | Modify |
| `src/blockchain_ai/train.py` | Modify |
| `scripts/run_pipeline.py` | Modify |
| `pyproject.toml` | Modify |
| `tests/test_config.py` | Create |
| `tests/test_ingest.py` | Modify |
| `tests/test_train.py` | Modify |

---

## Out of Scope

- Config validation beyond required key checks (no JSON Schema, Pydantic, etc.)
- Config inheritance or merging between files
- CLI override of individual config values
