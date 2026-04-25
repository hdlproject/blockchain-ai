# Regression Model Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scheduled regression model pipeline that ingests CSV data, trains a pluggable regression model, evaluates it, and saves benchmark reports.

**Architecture:** A flat pipeline of four focused modules (ingest → train → evaluate → predict) orchestrated by a single script. Each module reads/writes files to well-known directories so they can be run independently or chained together. The model algorithm is pluggable via a string argument.

**Tech Stack:** Python 3.13, pandas, scikit-learn, joblib

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/blockchain_ai/__init__.py` | Package marker |
| `src/blockchain_ai/ingest.py` | Load CSV, drop nulls, save to processed/ |
| `src/blockchain_ai/train.py` | Load processed CSV, train model, save to models/ |
| `src/blockchain_ai/evaluate.py` | Load model + processed CSV, compute metrics, save report |
| `src/blockchain_ai/predict.py` | Load model + input CSV, return predictions |
| `scripts/run_pipeline.py` | Orchestrate ingest → train → evaluate |
| `tests/test_ingest.py` | Unit tests for ingest |
| `tests/test_train.py` | Unit tests for train |
| `tests/test_evaluate.py` | Unit tests for evaluate |
| `tests/test_predict.py` | Unit tests for predict |
| `data/raw/.gitkeep` | Keep raw/ in git |
| `data/processed/.gitkeep` | Keep processed/ in git |
| `models/.gitkeep` | Keep models/ in git |
| `reports/.gitkeep` | Keep reports/ in git |
| `notebooks/.gitkeep` | Keep notebooks/ in git |

---

## Task 1: Project scaffold and dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `models/.gitkeep`, `reports/.gitkeep`, `notebooks/.gitkeep`
- Create: `src/blockchain_ai/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Replace the `dependencies = []` line in `pyproject.toml` with:

```toml
[project]
name = "blockchain-ai"
version = "0.1.0"
description = ""
authors = [
    {name = "Hendra Danu Laksana",email = "hdl.project.co@gmail.com"}
]
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
```

- [ ] **Step 2: Install dependencies**

```bash
poetry install --no-root
```

Expected: resolves and installs pandas, scikit-learn, joblib into the virtualenv.

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p data/raw data/processed models reports notebooks src/blockchain_ai tests scripts
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep reports/.gitkeep notebooks/.gitkeep
touch src/blockchain_ai/__init__.py tests/__init__.py
```

- [ ] **Step 4: Verify structure**

```bash
find . -not -path './.idea/*' -not -path './.claude/*' -not -path './docs/*' | sort
```

Expected output includes: `src/blockchain_ai/__init__.py`, `data/raw/`, `models/`, `reports/`

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml src/ tests/ data/ models/ reports/ notebooks/ scripts/
git commit -m "chore: scaffold project structure and add dependencies"
```

---

## Task 2: ingest module

**Files:**
- Create: `src/blockchain_ai/ingest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:

```python
import pandas as pd
import pytest
from pathlib import Path
from blockchain_ai.ingest import load_and_clean


def test_load_and_clean_drops_nulls(tmp_path):
    raw_csv = tmp_path / "input.csv"
    raw_csv.write_text("feature1,feature2,target\n1,2,3\n4,,6\n7,8,9\n")
    out_path = tmp_path / "processed.csv"

    load_and_clean(str(raw_csv), str(out_path))

    result = pd.read_csv(out_path)
    assert len(result) == 2
    assert list(result.columns) == ["feature1", "feature2", "target"]


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.csv"), str(tmp_path / "out.csv"))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement ingest.py**

Create `src/blockchain_ai/ingest.py`:

```python
import pandas as pd
from pathlib import Path


def load_and_clean(input_path: str, output_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(path)
    df = df.dropna()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/ingest.py tests/test_ingest.py
git commit -m "feat: add ingest module with null-dropping and file validation"
```

---

## Task 3: train module

**Files:**
- Create: `src/blockchain_ai/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import pandas as pd
import pytest
from pathlib import Path
from blockchain_ai.train import train_model


def test_train_model_saves_artifact(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 4, 6, 8, 10],
        "target": [3, 6, 9, 12, 15],
    })
    df.to_csv(csv_path, index=False)

    train_model(str(csv_path), "target", str(model_path), model_type="linear")

    assert model_path.exists()


def test_train_model_raises_on_unknown_model_type(tmp_path):
    csv_path = tmp_path / "processed.csv"
    df = pd.DataFrame({"feature1": [1, 2], "target": [3, 4]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(str(csv_path), "target", str(tmp_path / "m.joblib"), model_type="unknown")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_train.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement train.py**

Create `src/blockchain_ai/train.py`:

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression


def train_model(input_path: str, target_col: str, model_path: str, model_type: str = "linear") -> object:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Supported: 'linear'")

    model.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_train.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/train.py tests/test_train.py
git commit -m "feat: add train module with pluggable model type"
```

---

## Task 4: evaluate module

**Files:**
- Create: `src/blockchain_ai/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from blockchain_ai.evaluate import evaluate_model


def test_evaluate_model_saves_report(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    df.to_csv(csv_path, index=False)

    model = LinearRegression()
    model.fit(df[["feature1"]], df["target"])
    joblib.dump(model, model_path)

    evaluate_model(str(csv_path), "target", str(model_path), str(report_path))

    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "rmse" in report
    assert "mae" in report
    assert "r2" in report
    assert report["r2"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_model_report_values_are_floats(tmp_path):
    import pytest
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"

    df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "target": [1.5, 2.5, 3.5]})
    df.to_csv(csv_path, index=False)
    model = LinearRegression()
    model.fit(df[["feature1"]], df["target"])
    joblib.dump(model, model_path)

    evaluate_model(str(csv_path), "target", str(model_path), str(report_path))

    report = json.loads(report_path.read_text())
    assert isinstance(report["rmse"], float)
    assert isinstance(report["mae"], float)
    assert isinstance(report["r2"], float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_evaluate.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement evaluate.py**

Create `src/blockchain_ai/evaluate.py`:

```python
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(input_path: str, target_col: str, model_path: str, report_path: str) -> dict:
    df = pd.read_csv(input_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = joblib.load(model_path)
    predictions = model.predict(X)

    report = {
        "rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y, predictions)),
        "r2": float(r2_score(y, predictions)),
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_evaluate.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluate module with RMSE, MAE, R2 metrics"
```

---

## Task 5: predict module

**Files:**
- Create: `src/blockchain_ai/predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_predict.py`:

```python
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from blockchain_ai.predict import run_inference


def test_run_inference_returns_predictions(tmp_path):
    csv_path = tmp_path / "input.csv"
    model_path = tmp_path / "model.joblib"

    train_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    model = LinearRegression()
    model.fit(train_df[["feature1"]], train_df["target"])
    joblib.dump(model, model_path)

    input_df = pd.DataFrame({"feature1": [4.0, 5.0]})
    input_df.to_csv(csv_path, index=False)

    predictions = run_inference(str(csv_path), str(model_path))

    assert len(predictions) == 2
    assert np.isclose(predictions[0], 8.0, atol=0.1)
    assert np.isclose(predictions[1], 10.0, atol=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_predict.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement predict.py**

Create `src/blockchain_ai/predict.py`:

```python
import joblib
import pandas as pd
import numpy as np


def run_inference(input_path: str, model_path: str) -> np.ndarray:
    df = pd.read_csv(input_path)
    model = joblib.load(model_path)
    return model.predict(df)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_predict.py -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/predict.py tests/test_predict.py
git commit -m "feat: add predict module for running inference"
```

---

## Task 6: run_pipeline orchestration script

**Files:**
- Create: `scripts/run_pipeline.py`

- [ ] **Step 1: Implement run_pipeline.py**

Create `scripts/run_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Run the full regression pipeline: ingest -> train -> evaluate.

Usage:
    poetry run python scripts/run_pipeline.py \
        --raw data/raw/input.csv \
        --target target \
        [--model-type linear]
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
    parser.add_argument("--raw", required=True, help="Path to raw input CSV")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument("--model-type", default="linear", help="Model type (default: linear)")
    args = parser.parse_args()

    processed_path = "data/processed/processed.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/3] Ingesting {args.raw} ...")
    load_and_clean(args.raw, processed_path)

    print(f"[2/3] Training {args.model_type} model ...")
    train_model(processed_path, args.target, model_path, model_type=args.model_type)

    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(processed_path, args.target, model_path, report_path)

    print(f"\nPipeline complete. Report saved to {report_path}:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 3: Smoke test the pipeline with sample data**

```bash
cat > /tmp/sample.csv << 'EOF'
feature1,feature2,target
1,2,3
2,4,6
3,6,9
4,8,12
5,10,15
EOF

poetry run python scripts/run_pipeline.py --raw /tmp/sample.csv --target target
```

Expected output:
```
[1/3] Ingesting /tmp/sample.csv ...
[2/3] Training linear model ...
[3/3] Evaluating model ...

Pipeline complete. Report saved to reports/report.json:
  rmse: 0.0000
  mae: 0.0000
  r2: 1.0000
```

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat: add run_pipeline orchestration script"
```

---

## Task 7: Final verification

- [ ] **Step 1: Run full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: all tests pass, no warnings

- [ ] **Step 2: Verify project structure**

```bash
find . -not -path './.git/*' -not -path './.idea/*' -not -path './.claude/*' -not -path './docs/*' -not -path './__pycache__/*' -not -path './*/__pycache__/*' | sort
```

Expected to include:
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `models/.gitkeep`
- `reports/.gitkeep`
- `notebooks/.gitkeep`
- `src/blockchain_ai/__init__.py`
- `src/blockchain_ai/ingest.py`
- `src/blockchain_ai/train.py`
- `src/blockchain_ai/evaluate.py`
- `src/blockchain_ai/predict.py`
- `scripts/run_pipeline.py`
- `tests/test_ingest.py`
- `tests/test_train.py`
- `tests/test_evaluate.py`
- `tests/test_predict.py`
