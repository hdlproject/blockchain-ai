# Pipeline Config File Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded constants in `ingest.py` and `train.py` with a per-experiment YAML config file, making the pipeline fully configurable without code changes.

**Architecture:** A new `config.py` module loads YAML into typed dataclasses (`IngestConfig`, `TrainConfig`, `PipelineConfig`). `ingest.py` and `train.py` accept config objects instead of hardcoded values. `run_pipeline.py` takes a single `--config` flag and passes the loaded config through.

**Tech Stack:** Python 3.12, pyyaml, dataclasses (stdlib)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `configs/ethereum-gas-price.yaml` | Create | Ethereum gas price experiment config |
| `src/blockchain_ai/config.py` | Create | Load + validate YAML → typed dataclasses |
| `src/blockchain_ai/ingest.py` | Modify | Accept `IngestConfig` instead of hardcoded constants |
| `src/blockchain_ai/train.py` | Modify | Accept `TrainConfig` instead of hardcoded values |
| `scripts/run_pipeline.py` | Modify | `--config` flag replaces `--target` and `--model-type` |
| `pyproject.toml` | Modify | Add `pyyaml>=6.0` |
| `tests/test_config.py` | Create | Tests for `load_config` |
| `tests/test_ingest.py` | Modify | Pass `IngestConfig` to `load_and_clean` |
| `tests/test_train.py` | Modify | Pass `TrainConfig` to `train_model` |

---

## Task 1: Add pyyaml dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pyyaml to pyproject.toml**

Edit the `dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost (>=2.0.0)",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Install the dependency**

```bash
poetry add "pyyaml>=6.0"
```

Expected: installs pyyaml, `poetry.lock` updated.

- [ ] **Step 3: Verify import**

```bash
poetry run python -c "import yaml; print(yaml.__version__)"
```

Expected: prints a version string like `6.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add pyyaml dependency"
```

---

## Task 2: Create config.py and test_config.py

**Files:**
- Create: `src/blockchain_ai/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import pytest
from blockchain_ai.config import load_config, PipelineConfig, IngestConfig, TrainConfig


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return str(p)


_VALID_YAML = """
ingest:
  drop_cols:
    - hash
    - block_hash
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
    assert cfg.ingest.drop_cols == ["hash", "block_hash"]
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
  drop_cols: []
  fill_zero_cols: []
  timestamp_col: block_timestamp
  target_col: gas_price
"""
    path = _write_yaml(tmp_path, yaml)
    with pytest.raises(ValueError, match="train"):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'blockchain_ai.config'`

- [ ] **Step 3: Create config.py**

Create `src/blockchain_ai/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path
import yaml


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


def load_config(path: str) -> PipelineConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(p) as f:
        raw = yaml.safe_load(f)

    if "ingest" not in raw:
        raise ValueError("Config missing required key: 'ingest'")
    if "train" not in raw:
        raise ValueError("Config missing required key: 'train'")

    i = raw["ingest"]
    t = raw["train"]

    for key in ("drop_cols", "fill_zero_cols", "timestamp_col", "target_col"):
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    for key in ("target_col", "model_type", "stratify_col", "test_size", "hyperparameters"):
        if key not in t:
            raise ValueError(f"Config train section missing required key: '{key}'")

    return PipelineConfig(
        ingest=IngestConfig(
            drop_cols=i["drop_cols"],
            fill_zero_cols=i["fill_zero_cols"],
            timestamp_col=i["timestamp_col"],
            target_col=i["target_col"],
        ),
        train=TrainConfig(
            target_col=t["target_col"],
            model_type=t["model_type"],
            stratify_col=t["stratify_col"],
            test_size=t["test_size"],
            hyperparameters=t["hyperparameters"],
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/config.py tests/test_config.py
git commit -m "feat: add config module with YAML loading and typed dataclasses"
```

---

## Task 3: Create the default experiment config YAML

**Files:**
- Create: `configs/ethereum-gas-price.yaml`

- [ ] **Step 1: Create configs/ directory and YAML**

```bash
mkdir -p configs
```

Create `configs/ethereum-gas-price.yaml`:

```yaml
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

- [ ] **Step 2: Verify config loads cleanly**

```bash
poetry run python -c "
from blockchain_ai.config import load_config
cfg = load_config('configs/ethereum-gas-price.yaml')
print('ingest target_col:', cfg.ingest.target_col)
print('train model_type:', cfg.train.model_type)
print('n_estimators:', cfg.train.hyperparameters['n_estimators'])
"
```

Expected:
```
ingest target_col: gas_price
train model_type: xgboost
n_estimators: 300
```

- [ ] **Step 3: Commit**

```bash
git add configs/ethereum-gas-price.yaml
git commit -m "feat: add ethereum-gas-price experiment config"
```

---

## Task 4: Update ingest.py and test_ingest.py

**Files:**
- Modify: `src/blockchain_ai/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Update test_ingest.py to pass IngestConfig**

Replace `tests/test_ingest.py` entirely:

```python
import zipfile
import numpy as np
import pandas as pd
import pytest
from blockchain_ai.config import IngestConfig
from blockchain_ai.ingest import load_and_clean


def _make_zip(tmp_path, df):
    csv_path = tmp_path / "eth_transactions.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "ethereum-transactions.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(csv_path, "eth_transactions.csv")
    return str(zip_path)


def _minimal_df(**overrides):
    base = {
        "hash": ["0xabc"],
        "nonce": [1],
        "transaction_index": [0],
        "from_address": ["0x1"],
        "to_address": ["0x2"],
        "value": [1000.0],
        "gas": [21000],
        "gas_price": [12000000000],
        "input": ["0x"],
        "receipt_cumulative_gas_used": [21000],
        "receipt_gas_used": [21000],
        "receipt_contract_address": [None],
        "receipt_root": [None],
        "receipt_status": [1],
        "block_timestamp": ["2023-01-01 00:00:00+00:00"],
        "block_number": [16000000],
        "block_hash": ["0xdef"],
        "max_fee_per_gas": [None],
        "max_priority_fee_per_gas": [None],
        "transaction_type": [0],
        "receipt_effective_gas_price": [12000000000],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _default_config():
    return IngestConfig(
        drop_cols=["hash", "block_hash", "from_address", "to_address", "input",
                   "receipt_contract_address", "receipt_root"],
        fill_zero_cols=["max_fee_per_gas", "max_priority_fee_per_gas"],
        timestamp_col="block_timestamp",
        target_col="gas_price",
    )


def test_load_and_clean_drops_junk_columns(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    junk = {"hash", "block_hash", "from_address", "to_address", "input",
            "receipt_contract_address", "receipt_root", "gas_price"}
    assert not junk.intersection(result.columns)


def test_load_and_clean_fills_zero_cols(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    assert result["max_fee_per_gas"].iloc[0] == 0.0
    assert result["max_priority_fee_per_gas"].iloc[0] == 0.0


def test_load_and_clean_parses_block_timestamp(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    assert pd.api.types.is_integer_dtype(result["block_timestamp"])


def test_load_and_clean_adds_log_target(tmp_path):
    df = _minimal_df(gas_price=[12000000000])
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    assert "log_gas_price" in result.columns
    expected = np.log1p(12000000000)
    assert abs(result["log_gas_price"].iloc[0] - expected) < 1e-6


def test_load_and_clean_saves_csv(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    load_and_clean(zip_path, out_path, _default_config())

    assert (tmp_path / "out.csv").exists()


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.zip"), str(tmp_path / "out.csv"), _default_config())
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: FAIL — `load_and_clean` doesn't accept a `config` argument yet.

- [ ] **Step 3: Update ingest.py**

Replace `src/blockchain_ai/ingest.py` entirely:

```python
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from blockchain_ai.config import IngestConfig


def load_and_clean(input_path: str, output_path: str, config: IngestConfig) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(csv_name))

    df = df.drop(columns=[c for c in config.drop_cols if c in df.columns])
    df[config.fill_zero_cols] = df[config.fill_zero_cols].fillna(0.0)
    df[config.timestamp_col] = (
        pd.to_datetime(df[config.timestamp_col], utc=True).astype("int64") // 10**9
    )
    df[f"log_{config.target_col}"] = np.log1p(df[config.target_col])
    df = df.drop(columns=[config.target_col])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/ingest.py tests/test_ingest.py
git commit -m "feat: ingest accepts IngestConfig, removes hardcoded constants"
```

---

## Task 5: Update train.py and test_train.py

**Files:**
- Modify: `src/blockchain_ai/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Update test_train.py to pass TrainConfig**

Replace `tests/test_train.py` entirely:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_train.py -v
```

Expected: FAIL — `train_model` signature doesn't match yet.

- [ ] **Step 3: Update train.py**

Replace `src/blockchain_ai/train.py` entirely:

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from blockchain_ai.config import TrainConfig


def train_model(
    input_path: str,
    model_path: str,
    test_path: str,
    config: TrainConfig,
) -> object:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[config.target_col])
    y = df[config.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.hyperparameters.get("random_state", 42),
        stratify=df[config.stratify_col],
    )

    if config.model_type == "xgboost":
        model = XGBRegressor(**config.hyperparameters)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type!r}. Supported: 'xgboost'")

    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_df = X_test.copy()
    test_df[config.target_col] = y_test
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)

    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_train.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/train.py tests/test_train.py
git commit -m "feat: train accepts TrainConfig, hyperparameters driven by config"
```

---

## Task 6: Update run_pipeline.py

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Update run_pipeline.py**

Replace `scripts/run_pipeline.py` entirely:

```python
#!/usr/bin/env python3
"""
Run the full regression pipeline: ingest -> train -> evaluate.

Usage:
    poetry run python scripts/run_pipeline.py \
        --raw data/raw/ethereum-transactions.zip \
        --config configs/ethereum-gas-price.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.ingest import load_and_clean
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Run regression pipeline")
    parser.add_argument("--raw", required=True, help="Path to raw input zip")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_path = "data/processed/ethereum-transactions.csv"
    test_path = "data/processed/ethereum-transactions-test.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/3] Ingesting {args.raw} ...")
    load_and_clean(args.raw, processed_path, cfg.ingest)

    print(f"[2/3] Training {cfg.train.model_type} model ...")
    train_model(processed_path, model_path, test_path, cfg.train)

    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(test_path, cfg.train.target_col, model_path, report_path)

    print(f"\nPipeline complete. Report saved to {report_path}:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite**

```bash
poetry run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Smoke-test end-to-end**

```bash
poetry run python scripts/run_pipeline.py \
    --raw data/raw/ethereum-transactions.zip \
    --config configs/ethereum-gas-price.yaml
```

Expected output (approximate):
```
[1/3] Ingesting data/raw/ethereum-transactions.zip ...
[2/3] Training xgboost model ...
[3/3] Evaluating model ...

Pipeline complete. Report saved to reports/report.json:
  rmse: <large Wei value>
  mae: <large Wei value>
  r2: 0.9xxx
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat: pipeline takes --config flag, wires IngestConfig and TrainConfig end-to-end"
```
