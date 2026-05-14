# Optuna Hyperparameter Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Optuna-based HPO step to the pipeline that tunes XGBoost hyperparameters on training data and writes the best params back through the existing `TrainConfig` → `train_model` flow.

**Architecture:** A new `tune.py` module exposes a `run_hpo(input_path, config)` function that returns a `TrainConfig` with `hyperparameters` replaced by Optuna's best trial. An optional `hpo` section in the YAML config controls whether HPO runs and sets `n_trials`. `run_pipeline.py` runs HPO before training when the `hpo` section is present.

**Tech Stack:** `optuna>=3.0`, `xgboost>=2.0`, existing `TrainConfig` / `PipelineConfig` dataclasses, `pytest` for TDD.

---

## File Map

| File | Action |
|---|---|
| `src/blockchain_ai/tune.py` | Create |
| `src/blockchain_ai/config.py` | Modify — add `HpoConfig` dataclass and parse `hpo` section |
| `configs/ethereum-gas-price-predictor.yaml` | Modify — add optional `hpo` section |
| `scripts/run_pipeline.py` | Modify — run HPO when `cfg.hpo` is not None |
| `tests/test_tune.py` | Create |
| `tests/test_config.py` | Modify — add tests for `HpoConfig` loading |
| `pyproject.toml` | Modify — add `optuna>=3.0` dependency |

---

## Task 1: Add `optuna` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optuna to pyproject.toml**

In `pyproject.toml`, add to `dependencies`:
```toml
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost (>=2.0.0)",
    "pyyaml (>=6.0)",
    "optuna (>=3.0)",
]
```

- [ ] **Step 2: Install the dependency**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry add optuna`
Expected: optuna installed successfully

- [ ] **Step 3: Verify import works**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry run python -c "import optuna; print(optuna.__version__)"`
Expected: version string printed (e.g. `3.x.x`)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add optuna dependency"
```

---

## Task 2: Add `HpoConfig` to config module

**Files:**
- Modify: `src/blockchain_ai/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
def test_load_config_with_hpo_section(tmp_path):
    yaml_content = """
ingest:
  drop_cols: [hash]
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
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.hpo is not None
    assert cfg.hpo.n_trials == 20


def test_load_config_without_hpo_section(tmp_path):
    yaml_content = """
ingest:
  drop_cols: [hash]
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
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.hpo is None


def test_hpo_config_missing_n_trials_raises(tmp_path):
    yaml_content = """
ingest:
  drop_cols: [hash]
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
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    with pytest.raises(ValueError, match="hpo.*n_trials"):
        load_config(str(p))
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry run pytest tests/test_config.py -v -k "hpo" 2>&1 | tail -20`
Expected: FAIL — `HpoConfig` not defined, `cfg.hpo` attribute missing

- [ ] **Step 3: Add `HpoConfig` and update `PipelineConfig` and `load_config`**

In `src/blockchain_ai/config.py`, add after `TrainConfig`:

```python
@dataclass
class HpoConfig:
    n_trials: int
```

Update `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    hpo: HpoConfig | None = None
```

Update `load_config` — add after the `train` section validation block:

```python
    hpo_cfg = None
    if "hpo" in raw:
        h = raw["hpo"]
        if "n_trials" not in h:
            raise ValueError("Config hpo section missing required key: 'n_trials'")
        hpo_cfg = HpoConfig(n_trials=h["n_trials"])

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
        hpo=hpo_cfg,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry run pytest tests/test_config.py -v 2>&1 | tail -20`
Expected: all config tests pass (including 3 new hpo tests)

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/config.py tests/test_config.py
git commit -m "feat: add HpoConfig dataclass and optional hpo section loading"
```

---

## Task 3: Implement `tune.py`

**Files:**
- Create: `src/blockchain_ai/tune.py`
- Create: `tests/test_tune.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tune.py`:

```python
import pandas as pd
import pytest
from blockchain_ai.config import TrainConfig
from blockchain_ai.tune import run_hpo


def _processed_df():
    return pd.DataFrame({
        "nonce": list(range(40)),
        "transaction_index": list(range(40)),
        "value": [float(i * 1000) for i in range(40)],
        "gas": [21000] * 40,
        "receipt_cumulative_gas_used": [21000] * 40,
        "receipt_gas_used": [21000] * 40,
        "receipt_status": [1] * 40,
        "block_timestamp": [1672531200 + i * 12 for i in range(40)],
        "block_number": [16000000 + i for i in range(40)],
        "max_fee_per_gas": [0.0] * 40,
        "max_priority_fee_per_gas": [0.0] * 40,
        "transaction_type": [0] * 20 + [2] * 20,
        "receipt_effective_gas_price": [12000000000] * 40,
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
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry run pytest tests/test_tune.py -v 2>&1 | tail -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'blockchain_ai.tune'`

- [ ] **Step 3: Implement `tune.py`**

Create `src/blockchain_ai/tune.py`:

```python
import optuna
import pandas as pd
from dataclasses import replace
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBRegressor
from blockchain_ai.config import TrainConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hpo(input_path: str, config: TrainConfig, n_trials: int) -> TrainConfig:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[config.target_col])
    y = df[config.target_col]
    groups = df[config.stratify_col].astype(str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["random_state"] = 42

    return replace(config, hyperparameters=best_params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry run pytest tests/test_tune.py -v 2>&1 | tail -20`
Expected: all 4 tune tests pass

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/tune.py tests/test_tune.py
git commit -m "feat: add Optuna HPO module (tune.py)"
```

---

## Task 4: Wire HPO into `run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `configs/ethereum-gas-price-predictor.yaml`

- [ ] **Step 1: Update `run_pipeline.py` to call `run_hpo` when `cfg.hpo` is set**

Replace the existing `main()` in `scripts/run_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Run the full regression pipeline: ingest -> [hpo] -> train -> evaluate.

Usage:
    poetry run python scripts/run_pipeline.py \
        --raw data/raw/ethereum-transactions.zip \
        --config configs/ethereum-gas-price-predictor.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.ingest import load_and_clean
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model
from blockchain_ai.tune import run_hpo


def main():
    parser = argparse.ArgumentParser(description="Run regression pipeline")
    parser.add_argument("--raw", required=True, help="Path to raw input zip")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_path = "data/processed/ethereum-blocks.csv"
    test_path = "data/processed/ethereum-blocks-test.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/3] Ingesting {args.raw} ...")
    load_and_clean(args.raw, processed_path, cfg.ingest)

    train_config = cfg.train
    if cfg.hpo is not None:
        print(f"[2/3] Running Optuna HPO ({cfg.hpo.n_trials} trials) ...")
        train_config = run_hpo(processed_path, cfg.train, n_trials=cfg.hpo.n_trials)
        print(f"      Best hyperparameters: {train_config.hyperparameters}")
    else:
        print(f"[2/3] Skipping HPO (no hpo section in config)")

    print(f"[3/4] Training {train_config.model_type} model ...")
    train_model(processed_path, model_path, test_path, train_config)

    print(f"[4/4] Evaluating model ...")
    report = evaluate_model(test_path, train_config.target_col, model_path, report_path)

    print(f"\nPipeline complete. Report saved to {report_path}:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `hpo` section to the example config**

In `configs/ethereum-gas-price-predictor.yaml`, add after the `train` section:

```yaml
hpo:
  n_trials: 50
```

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/pintu/PycharmProjects/blockchain-ai && poetry run pytest -v 2>&1 | tail -30`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py configs/ethereum-gas-price-predictor.yaml
git commit -m "feat: wire Optuna HPO into pipeline (run when hpo config present)"
```

---

## Self-Review

**Spec coverage:**
- `optuna` dependency → Task 1 ✓
- `HpoConfig` dataclass → Task 2 ✓
- Optional `hpo` YAML section → Task 2 + Task 4 ✓
- `run_hpo()` function → Task 3 ✓
- Returns updated `TrainConfig` with best params → Task 3 ✓
- Pipeline integration (HPO before training) → Task 4 ✓
- Tests for all new behavior → Tasks 2, 3 ✓

**Placeholder scan:** None found. All steps have complete code.

**Type consistency:**
- `run_hpo(input_path: str, config: TrainConfig, n_trials: int) -> TrainConfig` — consistent across Task 3 (implementation) and Task 4 (call site)
- `HpoConfig.n_trials: int` — consistent across Task 2 (definition) and Task 4 (`cfg.hpo.n_trials`)
- `dataclasses.replace(config, hyperparameters=best_params)` — `TrainConfig` is a dataclass, `replace` is correct

**No gaps found.**
