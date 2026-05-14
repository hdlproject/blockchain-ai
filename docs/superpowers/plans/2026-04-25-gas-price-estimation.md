# Gas Price Estimation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Linear Regression pipeline with an XGBoost model that predicts Ethereum `gas_price` in log-space and reports metrics in Wei.

**Architecture:** Ingest extracts from a zip, drops junk columns, fills EIP-1559 nulls with 0, parses timestamps, and log-transforms the target. Train does an 80/20 stratified split, fits XGBRegressor, and saves both the model and the test split. Evaluate loads the test split, exponentiates predictions back to Wei, and writes RMSE/MAE/R² to a JSON report.

**Tech Stack:** Python 3.12, pandas, scikit-learn, xgboost, joblib, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `xgboost` dependency |
| `src/blockchain_ai/ingest.py` | Rewrite | Drop cols, fill nulls, parse timestamp, log-transform target, read from zip |
| `src/blockchain_ai/train.py` | Rewrite | XGBRegressor, 80/20 stratified split, save model + test split |
| `src/blockchain_ai/evaluate.py` | Modify | Load test split path, exponentiate predictions before metrics |
| `scripts/run_pipeline.py` | Modify | Update default `--model-type` to `"xgboost"`, pass test split path to evaluate |
| `tests/test_ingest.py` | Rewrite | Tests for new ingest behavior |
| `tests/test_train.py` | Rewrite | Tests for XGBoost training + test split output |
| `tests/test_evaluate.py` | Rewrite | Tests for expm1 evaluation on test split path |

---

## Task 1: Add xgboost dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add xgboost to pyproject.toml**

Open `pyproject.toml` and add `xgboost` to the `[project]` dependencies list:

```toml
[project]
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost>=2.0.0",
]
```

- [ ] **Step 2: Install the new dependency**

```bash
poetry install
```

Expected: installs xgboost without errors, `poetry.lock` updated.

- [ ] **Step 3: Verify import works**

```bash
python -c "import xgboost; print(xgboost.__version__)"
```

Expected: prints a version string like `2.x.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add xgboost dependency"
```

---

## Task 2: Rewrite ingest.py

**Files:**
- Modify: `src/blockchain_ai/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_ingest.py` entirely:

```python
import zipfile
import numpy as np
import pandas as pd
import pytest
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


def test_load_and_clean_drops_junk_columns(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path)

    junk = {"hash", "block_hash", "from_address", "to_address", "input",
            "receipt_contract_address", "receipt_root", "gas_price"}
    assert not junk.intersection(result.columns)


def test_load_and_clean_fills_eip1559_nulls(tmp_path):
    df = _minimal_df()  # max_fee_per_gas and max_priority_fee_per_gas are None
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path)

    assert result["max_fee_per_gas"].iloc[0] == 0.0
    assert result["max_priority_fee_per_gas"].iloc[0] == 0.0


def test_load_and_clean_parses_block_timestamp(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path)

    assert pd.api.types.is_integer_dtype(result["block_timestamp"])


def test_load_and_clean_adds_log_gas_price(tmp_path):
    df = _minimal_df(gas_price=[12000000000])
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path)

    assert "log_gas_price" in result.columns
    expected = np.log1p(12000000000)
    assert abs(result["log_gas_price"].iloc[0] - expected) < 1e-6


def test_load_and_clean_saves_csv(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    load_and_clean(zip_path, out_path)

    assert (tmp_path / "out.csv").exists()


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.zip"), str(tmp_path / "out.csv"))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingest.py -v
```

Expected: multiple FAILs — current ingest reads CSV not zip, does blanket dropna, no log transform.

- [ ] **Step 3: Rewrite ingest.py**

Replace `src/blockchain_ai/ingest.py` entirely:

```python
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

_DROP_COLS = [
    "hash", "block_hash", "from_address", "to_address", "input",
    "receipt_contract_address", "receipt_root",
]
_EIP1559_COLS = ["max_fee_per_gas", "max_priority_fee_per_gas"]


def load_and_clean(input_path: str, output_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(csv_name))

    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    df[_EIP1559_COLS] = df[_EIP1559_COLS].fillna(0.0)
    df["block_timestamp"] = (
        pd.to_datetime(df["block_timestamp"], utc=True).astype("int64") // 10**9
    )
    df["log_gas_price"] = np.log1p(df["gas_price"])
    df = df.drop(columns=["gas_price"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ingest.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/ingest.py tests/test_ingest.py
git commit -m "feat: rewrite ingest to read zip, log-transform gas_price, fill EIP-1559 nulls"
```

---

## Task 3: Rewrite train.py

**Files:**
- Modify: `src/blockchain_ai/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_train.py` entirely:

```python
import pandas as pd
import pytest
import joblib
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


def test_train_model_saves_model_artifact(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)

    train_model(str(csv_path), "log_gas_price", str(model_path), str(test_path))

    assert model_path.exists()


def test_train_model_saves_test_split(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)

    train_model(str(csv_path), "log_gas_price", str(model_path), str(test_path))

    assert test_path.exists()
    test_df = pd.read_csv(test_path)
    assert "log_gas_price" in test_df.columns
    assert len(test_df) == 4  # 20% of 20 rows


def test_train_model_raises_on_unknown_model_type(tmp_path):
    csv_path = tmp_path / "processed.csv"
    _processed_df().to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(
            str(csv_path), "log_gas_price",
            str(tmp_path / "m.joblib"), str(tmp_path / "t.csv"),
            model_type="unknown",
        )


def test_train_model_default_is_xgboost(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df().to_csv(csv_path, index=False)

    train_model(str(csv_path), "log_gas_price", str(model_path), str(test_path))

    model = joblib.load(model_path)
    assert "XGB" in type(model).__name__
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train.py -v
```

Expected: FAILs — old `train_model` signature doesn't accept `test_path`, uses LinearRegression.

- [ ] **Step 3: Rewrite train.py**

Replace `src/blockchain_ai/train.py` entirely:

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_model(
    input_path: str,
    target_col: str,
    model_path: str,
    test_path: str,
    model_type: str = "xgboost",
) -> object:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df["transaction_type"]
    )

    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Supported: 'xgboost'")

    model.fit(X_train, y_train)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    test_df = X_test.copy()
    test_df[target_col] = y_test
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(test_path, index=False)

    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/train.py tests/test_train.py
git commit -m "feat: replace LinearRegression with XGBRegressor, save test split"
```

---

## Task 4: Update evaluate.py

**Files:**
- Modify: `src/blockchain_ai/evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_evaluate.py` entirely:

```python
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor
from blockchain_ai.evaluate import evaluate_model


def _make_test_split(tmp_path):
    """Creates a test CSV where log_gas_price is perfectly predictable."""
    X = pd.DataFrame({
        "nonce": list(range(10)),
        "transaction_index": list(range(10)),
        "value": [float(i * 1000) for i in range(10)],
        "gas": [21000] * 10,
        "receipt_cumulative_gas_used": [21000] * 10,
        "receipt_gas_used": [21000] * 10,
        "receipt_status": [1] * 10,
        "block_timestamp": [1672531200 + i * 12 for i in range(10)],
        "block_number": [16000000 + i for i in range(10)],
        "max_fee_per_gas": [0.0] * 10,
        "max_priority_fee_per_gas": [0.0] * 10,
        "transaction_type": [0] * 5 + [2] * 5,
        "receipt_effective_gas_price": [12000000000] * 10,
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


def test_evaluate_model_saves_report(tmp_path):
    test_path, model_path = _make_test_split(tmp_path)
    report_path = str(tmp_path / "report.json")

    evaluate_model(test_path, "log_gas_price", model_path, report_path)

    assert (tmp_path / "report.json").exists()
    report = json.loads((tmp_path / "report.json").read_text())
    assert "rmse" in report
    assert "mae" in report
    assert "r2" in report


def test_evaluate_model_report_values_are_floats(tmp_path):
    test_path, model_path = _make_test_split(tmp_path)
    report_path = str(tmp_path / "report.json")

    evaluate_model(test_path, "log_gas_price", model_path, report_path)

    report = json.loads((tmp_path / "report.json").read_text())
    assert isinstance(report["rmse"], float)
    assert isinstance(report["mae"], float)
    assert isinstance(report["r2"], float)


def test_evaluate_model_metrics_are_in_wei(tmp_path):
    """RMSE should be in Wei (large numbers), not log-space (small numbers ~23)."""
    test_path, model_path = _make_test_split(tmp_path)
    report_path = str(tmp_path / "report.json")

    evaluate_model(test_path, "log_gas_price", model_path, report_path)

    report = json.loads((tmp_path / "report.json").read_text())
    # expm1(23.2) ≈ 1.2e10 Wei — RMSE in log-space would be < 1
    assert report["rmse"] > 1.0 or report["rmse"] == pytest.approx(0.0, abs=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evaluate.py -v
```

Expected: FAILs — current `evaluate_model` signature has no `test_path` concept and doesn't exponentiate.

- [ ] **Step 3: Update evaluate.py**

Replace `src/blockchain_ai/evaluate.py` entirely:

```python
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(
    test_path: str, target_col: str, model_path: str, report_path: str
) -> dict:
    df = pd.read_csv(test_path)
    X = df.drop(columns=[target_col])
    y_log = df[target_col]

    model = joblib.load(model_path)
    y_pred_log = model.predict(X)

    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    report = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate on test split, report metrics in Wei via expm1"
```

---

## Task 5: Update run_pipeline.py

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
        --target log_gas_price \
        [--model-type xgboost]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.ingest import load_and_clean
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Run regression pipeline")
    parser.add_argument("--raw", required=True, help="Path to raw input zip")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument("--model-type", default="xgboost", help="Model type (default: xgboost)")
    args = parser.parse_args()

    processed_path = "data/processed/ethereum-blocks.csv"
    test_path = "data/processed/ethereum-blocks-test.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/3] Ingesting {args.raw} ...")
    load_and_clean(args.raw, processed_path)

    print(f"[2/3] Training {args.model_type} model ...")
    train_model(processed_path, args.target, model_path, test_path, model_type=args.model_type)

    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(test_path, args.target, model_path, report_path)

    print(f"\nPipeline complete. Report saved to {report_path}:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

```bash
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 3: Smoke-test the full pipeline end-to-end**

```bash
poetry run python scripts/run_pipeline.py \
    --raw data/raw/ethereum-transactions.zip \
    --target log_gas_price
```

Expected output (approximate):
```
[1/3] Ingesting data/raw/ethereum-transactions.zip ...
[2/3] Training xgboost model ...
[3/3] Evaluating model ...

Pipeline complete. Report saved to reports/report.json:
  rmse: <large number in Wei>
  mae: <large number in Wei>
  r2: <value between 0 and 1>
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat: wire up xgboost pipeline end-to-end, update default model-type"
```
