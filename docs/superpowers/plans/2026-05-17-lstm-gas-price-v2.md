# LSTM Gas Price Predictor v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LSTM-based gas price predictor served at `GET /predict/gas-price/v2/latest`, keeping all existing XGBoost v1 code untouched.

**Architecture:** A sklearn-compatible `LSTMWrapper` wraps a PyTorch `_LSTMNet`. It fits on sliding windows of shape `(n, seq_len, n_features)` and predicts via vectorised forward pass with zero-padding. A new FastAPI router handles auto-regressive multi-step forecasting. `app.py` mounts the v2 router only when `CONFIG_V2` env var is set.

**Tech Stack:** Python 3.12, PyTorch ≥ 2.0, FastAPI, joblib, pandas, numpy, pytest

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `src/blockchain_ai/model/__init__.py` | CREATE | Empty package marker |
| `src/blockchain_ai/model/lstm.py` | CREATE | `_LSTMNet` + `LSTMWrapper` |
| `configs/ethereum-gas-price-predictor-v2.yaml` | CREATE | v2 model config (lstm hyperparams) |
| `src/blockchain_ai/server/router_gas_price_v2.py` | CREATE | `GET /predict/gas-price/v2/latest` |
| `src/blockchain_ai/train.py` | MODIFY | Add `elif model_type == "lstm"` branch |
| `app.py` | MODIFY | Mount v2 router when `CONFIG_V2` is set |
| `tests/test_lstm.py` | CREATE | Unit tests for `LSTMWrapper` |
| `tests/test_router_gas_price_v2.py` | CREATE | Router tests with mocked model + client |
| `pyproject.toml` | MODIFY | Add `torch>=2.0.0` dependency |

---

## Task 1: Add PyTorch dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add torch to pyproject.toml dependencies**

Open `pyproject.toml` and add `"torch (>=2.0.0)"` to the `dependencies` list so it reads:

```toml
dependencies = [
    "pandas>=2.0.0",
    "scikit-learn>=1.4.0",
    "joblib>=1.3.0",
    "xgboost (>=2.0.0)",
    "pyyaml (>=6.0)",
    "optuna (>=3.0)",
    "plotext (>=5.3.2,<6.0.0)",
    "fastapi (>=0.111.0)",
    "uvicorn[standard] (>=0.29.0)",
    "python-multipart (>=0.0.9)",
    "requests (>=2.33.1,<3.0.0)",
    "python-dotenv (>=1.2.2,<2.0.0)",
    "google-cloud-storage (>=2.0.0)",
    "streamlit (>=1.35.0)",
    "torch (>=2.0.0)",
]
```

- [ ] **Step 2: Install the dependency**

```bash
poetry add torch
```

Expected: Poetry resolves and installs torch; `poetry.lock` is updated.

- [ ] **Step 3: Verify import**

```bash
python -c "import torch; print(torch.__version__)"
```

Expected: prints a version string like `2.x.x`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: add torch dependency for LSTM model"
```

---

## Task 2: Write failing LSTM tests

**Files:**
- Create: `tests/test_lstm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lstm.py` with the full content below:

```python
import numpy as np
import pandas as pd
import pytest
import joblib
from blockchain_ai.model.lstm import LSTMWrapper

_N = 60
_FEATURES = 3
_SEQ_LEN = 5


def _data(n=_N, features=_FEATURES, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, features)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1]
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(features)]), pd.Series(y)


def test_fit_predict_returns_finite_array():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=3, hidden_size=8, num_layers=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_predict_shape_matches_input_length():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    assert model.predict(X).shape == (len(X),)


def test_predict_zero_padding_for_short_input():
    """predict() on fewer rows than seq_len must still return that many predictions."""
    X, y = _data(n=40)
    model = LSTMWrapper(sequence_length=10, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    short = X.iloc[:3]
    preds = model.predict(short)
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()


def test_predict_accepts_numpy_array():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    preds = model.predict(X.values)
    assert preds.shape == (len(X),)


def test_feature_cols_stored_after_fit():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    assert model.feature_cols == ["f0", "f1", "f2"]


def test_early_stopping_does_not_crash():
    """Training completes without error even if early stopping fires."""
    X, y = _data(n=100)
    model = LSTMWrapper(
        sequence_length=_SEQ_LEN, epochs=50, hidden_size=4, num_layers=1, dropout=0.0
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_joblib_round_trip_preserves_predictions(tmp_path):
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=3, hidden_size=8, num_layers=1)
    model.fit(X, y)
    before = model.predict(X)

    path = tmp_path / "lstm.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    after = loaded.predict(X)

    np.testing.assert_array_almost_equal(before, after, decimal=5)


def test_sequence_length_attribute_accessible():
    model = LSTMWrapper(sequence_length=15)
    assert model.sequence_length == 15
```

- [ ] **Step 2: Run tests — verify they all fail**

```bash
pytest tests/test_lstm.py -v
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'blockchain_ai.model'`.

---

## Task 3: Implement model package

**Files:**
- Create: `src/blockchain_ai/model/__init__.py`
- Create: `src/blockchain_ai/model/lstm.py`

- [ ] **Step 1: Create the package marker**

Create `src/blockchain_ai/model/__init__.py` as an empty file.

- [ ] **Step 2: Implement lstm.py**

Create `src/blockchain_ai/model/lstm.py`:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMWrapper:
    """Sklearn-compatible wrapper around a stacked LSTM regressor."""

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.feature_cols: list[str] = []
        self._net: _LSTMNet | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y) -> "LSTMWrapper":
        if isinstance(X, pd.DataFrame):
            self.feature_cols = list(X.columns)
            X = X.values
        X = X.astype(np.float32)
        y_arr = np.array(y.values if hasattr(y, "values") else y, dtype=np.float32)

        n, n_features = X.shape
        seqs, targets = [], []
        for i in range(self.sequence_length, n):
            seqs.append(X[i - self.sequence_length : i])
            targets.append(y_arr[i])
        seqs = np.array(seqs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        split = max(1, int(len(seqs) * 0.9))
        X_train, X_val = seqs[:split], seqs[split:]
        y_train, y_val = targets[:split], targets[split:]

        self._net = _LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        patience = 0

        for _ in range(self.epochs):
            self._net.train()
            perm = np.random.permutation(len(X_train))
            for start in range(0, len(X_train), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.from_numpy(X_train[idx])
                yb = torch.from_numpy(y_train[idx])
                optimizer.zero_grad()
                loss_fn(self._net(xb), yb).backward()
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                val_loss = loss_fn(
                    self._net(torch.from_numpy(X_val)),
                    torch.from_numpy(y_val),
                ).item()

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float32)
        n, n_features = X.shape

        sequences = np.zeros((n, self.sequence_length, n_features), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.sequence_length + 1)
            seq = X[start : i + 1]
            sequences[i, self.sequence_length - len(seq) :] = seq

        self._net.eval()
        with torch.no_grad():
            return self._net(torch.from_numpy(sequences)).numpy()
```

- [ ] **Step 3: Run tests — verify they all pass**

```bash
pytest tests/test_lstm.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/blockchain_ai/model/ tests/test_lstm.py
git commit -m "feat: add LSTMWrapper for sequence-based gas price prediction"
```

---

## Task 4: Create v2 config file

**Files:**
- Create: `configs/ethereum-gas-price-predictor-v2.yaml`

- [ ] **Step 1: Create the config**

Create `configs/ethereum-gas-price-predictor-v2.yaml`:

```yaml
etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

collect:
  n_blocks: 2000
  output_path: data/raw/ethereum-blocks.csv

paths:
  processed_path: data/processed/ethereum-blocks.csv
  test_path: data/processed/ethereum-blocks-test.csv
  report_path: reports/gas_price_predictor_v2.json

ingest:
  feature_cols:
    - base_fee_gwei
    - gas_used_ratio
    - hour_of_day
    - day_of_week
    - base_fee_trend
  fill_zero_cols: []
  target_col: log_next_base_fee_gwei

train:
  target_col: log_next_base_fee_gwei
  model_type: lstm
  test_size: 0.2
  hyperparameters:
    sequence_length: 30
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    epochs: 50
    lr: 0.001
    batch_size: 32

serve:
  title: Ethereum Base Fee Predictor v2 (LSTM)
  description: >
    Predicts the next block's base fee using an LSTM sequence model trained on
    recent block history. Requires sequence context — no single-row POST endpoint.
  model_path: models/gas_price_predictor_v2.joblib
  target_description: Predicted next-block base fee
  target_unit: Gwei
  log_transform: true
  fields: {}
```

- [ ] **Step 2: Verify config loads without error**

```bash
python -c "
from blockchain_ai.config import load_config
cfg = load_config('configs/ethereum-gas-price-predictor-v2.yaml')
print('task:', cfg.task)
print('model_type:', cfg.train.model_type)
print('model_path:', cfg.serve.model_path)
print('hyperparameters:', cfg.train.hyperparameters)
"
```

Expected output:
```
task: regression
model_type: lstm
model_path: models/gas_price_predictor_v2.joblib
hyperparameters: {'sequence_length': 30, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'epochs': 50, 'lr': 0.001, 'batch_size': 32}
```

- [ ] **Step 3: Commit**

```bash
git add configs/ethereum-gas-price-predictor-v2.yaml
git commit -m "feat: add v2 LSTM config for gas price predictor"
```

---

## Task 5: Add lstm branch to train.py

**Files:**
- Modify: `src/blockchain_ai/train.py`
- Modify: `tests/test_train.py` (add one test case)

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_train.py`. First, add `import joblib` at the top if not present (it's already there). Add this function at the bottom of the file:

```python
def test_train_model_lstm_saves_model_and_has_predict(tmp_path):
    n = 50
    feature_cols = ["base_fee_gwei", "gas_used_ratio", "hour_of_day", "day_of_week", "base_fee_trend"]
    data = pd.DataFrame({
        "base_fee_gwei": [15.0 + i * 0.01 for i in range(n)],
        "gas_used_ratio": [0.5] * n,
        "hour_of_day": [i % 24 for i in range(n)],
        "day_of_week": [i % 7 for i in range(n)],
        "base_fee_trend": [0.01] * n,
        "log_next_base_fee_gwei": [2.7 + i * 0.001 for i in range(n)],
    })
    csv_path = tmp_path / "blocks.csv"
    model_path = tmp_path / "model.joblib"
    data.to_csv(csv_path, index=False)
    train_model(
        str(csv_path),
        str(model_path),
        str(tmp_path / "test.csv"),
        _cfg(
            "log_next_base_fee_gwei",
            model_type="lstm",
            hyperparameters={
                "sequence_length": 5,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "epochs": 2,
                "lr": 0.01,
                "batch_size": 4,
            },
        ),
    )
    assert model_path.exists()
    loaded = joblib.load(model_path)
    assert hasattr(loaded, "predict")
    assert hasattr(loaded, "sequence_length")
```

- [ ] **Step 2: Run the failing test**

```bash
pytest tests/test_train.py::test_train_model_lstm_saves_model_and_has_predict -v
```

Expected: FAIL with `ValueError: Unknown model_type: 'lstm'`.

- [ ] **Step 3: Add the lstm branch to train.py**

In `src/blockchain_ai/train.py`, find the `elif cfg.task == "regression":` block and replace:

```python
    elif cfg.task == "regression":
        if train_config.model_type == "xgboost":
            model = XGBRegressor(**train_config.hyperparameters)
        else:
            raise ValueError(f"Unknown model_type: {train_config.model_type!r}. Supported: 'xgboost'")
```

With:

```python
    elif cfg.task == "regression":
        if train_config.model_type == "xgboost":
            model = XGBRegressor(**train_config.hyperparameters)
        elif train_config.model_type == "lstm":
            from blockchain_ai.model.lstm import LSTMWrapper
            model = LSTMWrapper(**train_config.hyperparameters)
        else:
            raise ValueError(
                f"Unknown model_type: {train_config.model_type!r}. Supported: 'xgboost', 'lstm'"
            )
```

- [ ] **Step 4: Run all train tests**

```bash
pytest tests/test_train.py -v
```

Expected: all tests PASS (including the new lstm test).

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/train.py tests/test_train.py
git commit -m "feat: add lstm branch to train_model"
```

---

## Task 6: Write failing router v2 tests

**Files:**
- Create: `tests/test_router_gas_price_v2.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_router_gas_price_v2.py`:

```python
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from blockchain_ai.config import ServeConfig
from blockchain_ai.server.router_gas_price_v2 import create_router

_FEATURE_COLS = ["base_fee_gwei", "gas_used_ratio", "hour_of_day", "day_of_week", "base_fee_trend"]
_SEQ_LEN = 5

_SERVE = ServeConfig(
    model_path="models/gas_price_predictor_v2.joblib",
    title="v2",
    description="test",
    target_description="base fee",
    target_unit="Gwei",
    log_transform=False,
    fields={},
)


def _make_rows(n=40, start_fee=15.0):
    return [
        {
            "block_number": 21_000_000 + i,
            "base_fee_per_gas": int((start_fee + i * 0.01) * 1e9),
            "gas_used_ratio": 0.5,
            "timestamp": 1_700_000_000 + i * 12,
        }
        for i in range(n)
    ]


def _make_model(seq_len=_SEQ_LEN, pred=16.0):
    model = MagicMock()
    model.sequence_length = seq_len
    model.predict.return_value = np.full(seq_len, pred, dtype=np.float32)
    return model


def _make_client(n_rows=40):
    rows = _make_rows(n_rows)
    client = MagicMock()
    client.get_latest_block_number.return_value = rows[-1]["block_number"]
    client.get_block.side_effect = lambda n: next(
        (r for r in rows if r["block_number"] == n), None
    )
    return client


def _app(model=None, client=None, serve=_SERVE):
    app = FastAPI()
    app.include_router(create_router(serve, _FEATURE_COLS, model, client))
    return TestClient(app)


def test_predict_v2_returns_200():
    resp = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest")
    assert resp.status_code == 200


def test_predict_v2_response_shape():
    data = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest").json()
    assert "block_number" in data
    assert "block_history" in data
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    p = data["predictions"][0]
    assert set(p.keys()) == {"step", "block_number", "base_fee_gwei", "base_fee_wei", "method"}
    assert p["method"] == "lstm"
    assert p["step"] == 1


def test_predict_v2_n_blocks_2_returns_2_predictions():
    data = _app(_make_model(), _make_client()).get("/predict/gas-price/v2/latest?n_blocks=2").json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["step"] == 1
    assert data["predictions"][1]["step"] == 2


def test_predict_v2_no_model_returns_503():
    resp = _app(model=None, client=_make_client()).get("/predict/gas-price/v2/latest")
    assert resp.status_code == 503


def test_predict_v2_no_client_returns_503():
    resp = _app(model=_make_model(), client=None).get("/predict/gas-price/v2/latest")
    assert resp.status_code == 503


def test_predict_v2_log_transform_applied():
    serve = ServeConfig(
        model_path="m.joblib",
        title="v",
        description="d",
        target_description="fee",
        target_unit="Gwei",
        log_transform=True,
        fields={},
    )
    raw_pred = 2.0
    model = MagicMock()
    model.sequence_length = _SEQ_LEN
    model.predict.return_value = np.full(_SEQ_LEN, raw_pred, dtype=np.float32)

    data = _app(model, _make_client(), serve).get("/predict/gas-price/v2/latest").json()
    expected = float(np.expm1(raw_pred))
    assert abs(data["predictions"][0]["base_fee_gwei"] - expected) < 1e-4


def test_predict_v2_block_number_is_last_historical():
    client = _make_client()
    data = _app(_make_model(), client).get("/predict/gas-price/v2/latest").json()
    assert data["block_number"] == 21_000_039
    assert data["predictions"][0]["block_number"] == 21_000_040
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_router_gas_price_v2.py -v
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'blockchain_ai.server.router_gas_price_v2'`.

---

## Task 7: Implement router_gas_price_v2.py

**Files:**
- Create: `src/blockchain_ai/server/router_gas_price_v2.py`

- [ ] **Step 1: Implement the router**

Create `src/blockchain_ai/server/router_gas_price_v2.py`:

```python
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from blockchain_ai.feature.block_features import BlockFeatureExtractor
from blockchain_ai.config import ServeConfig


def create_router(
    serve: ServeConfig,
    feature_cols: list[str],
    model,
    etherscan_client,
) -> APIRouter:
    router = APIRouter()

    def _fetch_blocks() -> pd.DataFrame:
        if etherscan_client is None:
            raise HTTPException(
                status_code=503,
                detail="Etherscan client not available. Check ETHERSCAN_API_KEY.",
            )
        n_fetch = model.sequence_length + BlockFeatureExtractor.TREND_LOOKBACK
        latest = etherscan_client.get_latest_block_number()
        raw_rows = [
            etherscan_client.get_block(n)
            for n in range(latest - n_fetch + 1, latest + 1)
        ]
        raw_rows = [r for r in raw_rows if r]
        try:
            return BlockFeatureExtractor().extract(raw_rows)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    def _predict_one(rolling: pd.DataFrame) -> float:
        raw = float(model.predict(rolling)[-1])
        return float(np.expm1(raw) if serve.log_transform else raw)

    @router.get(
        "/predict/gas-price/v2/latest",
        summary="Predict next-block base fee using LSTM (v2)",
        description=(
            "Fetches recent blocks from Etherscan, feeds the last sequence_length rows "
            "into the LSTM, and returns up to n_blocks auto-regressive predictions."
        ),
    )
    def predict_latest_v2(n_blocks: int = Query(default=1, ge=1, le=50)):
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available yet. Run training pipeline first.",
            )
        df = _fetch_blocks()
        seq_len = model.sequence_length
        rolling_df = df[feature_cols].tail(seq_len).reset_index(drop=True)
        last = df.iloc[-1]
        last_block = int(last["block_number"])
        last_timestamp = float(last["timestamp"])
        gas_used_ratio = float(last["gas_used_ratio"])

        predictions = []
        for step in range(1, n_blocks + 1):
            pred_gwei = _predict_one(rolling_df)
            timestamp = last_timestamp + step * 12
            predictions.append({
                "step": step,
                "block_number": last_block + step,
                "base_fee_gwei": pred_gwei,
                "base_fee_wei": pred_gwei * 1e9,
                "method": "lstm",
            })
            if step < n_blocks:
                dt = pd.Timestamp(timestamp, unit="s", tz="UTC")
                ref_fee = float(
                    rolling_df["base_fee_gwei"].iloc[-BlockFeatureExtractor.TREND_LOOKBACK]
                )
                trend = (pred_gwei - ref_fee) / ref_fee if ref_fee > 0 else 0.0
                new_row = pd.DataFrame([{
                    "base_fee_gwei": pred_gwei,
                    "gas_used_ratio": gas_used_ratio,
                    "hour_of_day": dt.hour,
                    "day_of_week": dt.dayofweek,
                    "base_fee_trend": trend,
                }])[feature_cols]
                rolling_df = pd.concat(
                    [rolling_df.iloc[1:], new_row], ignore_index=True
                )

        return {
            "block_number": last_block,
            "block_history": (
                df[["block_number", "base_fee_gwei"]]
                .rename(columns={"block_number": "block"})
                .to_dict(orient="records")
            ),
            "predictions": predictions,
        }

    return router
```

- [ ] **Step 2: Run the router tests**

```bash
pytest tests/test_router_gas_price_v2.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 3: Run the full test suite to check for regressions**

```bash
pytest --ignore=tests/test_streamlit_ui.py --ignore=tests/test_collect.py -v
```

Expected: all tests PASS (test_streamlit_ui.py and test_collect.py require live network access).

- [ ] **Step 4: Commit**

```bash
git add src/blockchain_ai/server/router_gas_price_v2.py tests/test_router_gas_price_v2.py
git commit -m "feat: add LSTM gas price router v2 at /predict/gas-price/v2/latest"
```

---

## Task 8: Mount v2 router in app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add CONFIG_V2 block to app.py**

Open `app.py`. After the existing `elif task == "classification":` block (currently the last block in the file), append the following:

```python
_CONFIG_V2_PATH = os.environ.get("CONFIG_V2")
if _CONFIG_V2_PATH:
    _cfg_v2 = load_config(_CONFIG_V2_PATH)
    if _cfg_v2.serve is None:
        print(f"WARNING: CONFIG_V2 config at {_CONFIG_V2_PATH} has no 'serve' section — v2 router skipped.")
    else:
        _raw_model_path_v2 = os.environ.get("MODEL_PATH_V2", _cfg_v2.serve.model_path)
        model_v2 = None
        try:
            if _raw_model_path_v2.startswith("gs://"):
                import tempfile
                from google.cloud import storage as gcs
                _tmp_v2 = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
                _bucket_v2, _, _blob_v2 = _raw_model_path_v2[5:].partition("/")
                gcs.Client().bucket(_bucket_v2).blob(_blob_v2).download_to_filename(_tmp_v2.name)
                _model_path_v2 = _tmp_v2.name
            else:
                _model_path_v2 = _raw_model_path_v2
            model_v2 = joblib.load(_model_path_v2)
        except Exception as exc:
            print(f"WARNING: v2 model could not be loaded ({exc}). v2 endpoint will return 503.")
        from blockchain_ai.server.router_gas_price_v2 import create_router as create_router_v2
        app.include_router(
            create_router_v2(
                _cfg_v2.serve,
                _cfg_v2.ingest.feature_cols,
                model_v2,
                _etherscan_client,
            )
        )
```

- [ ] **Step 2: Smoke-test that the v1 server still starts without CONFIG_V2**

```bash
CONFIG=configs/ethereum-gas-price-predictor.yaml python -c "import app; print('app loaded ok')"
```

Expected: prints `app loaded ok` (with a WARNING about the missing model file — that's normal in a dev environment without a trained model).

- [ ] **Step 3: Smoke-test that both routers load when CONFIG_V2 is set**

First create a placeholder model so the v2 router doesn't 503 at startup:

```bash
python -c "
import joblib, numpy as np, sys
sys.path.insert(0, 'src')
from blockchain_ai.model.lstm import LSTMWrapper
import pandas as pd
model = LSTMWrapper(sequence_length=5, hidden_size=8, num_layers=1, epochs=1)
X = pd.DataFrame(np.random.rand(10, 5), columns=['base_fee_gwei','gas_used_ratio','hour_of_day','day_of_week','base_fee_trend'])
y = pd.Series(np.random.rand(10))
model.fit(X, y)
import os; os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/gas_price_predictor_v2.joblib')
print('placeholder v2 model saved')
"

CONFIG=configs/ethereum-gas-price-predictor.yaml \
CONFIG_V2=configs/ethereum-gas-price-predictor-v2.yaml \
python -c "import app; print('both routers loaded ok')"
```

Expected: prints `both routers loaded ok`.

- [ ] **Step 4: Verify routes are registered**

```bash
CONFIG=configs/ethereum-gas-price-predictor.yaml \
CONFIG_V2=configs/ethereum-gas-price-predictor-v2.yaml \
python -c "
import app
routes = [r.path for r in app.app.routes]
print(routes)
assert '/predict/gas-price/latest' in routes, 'v1 route missing'
assert '/predict/gas-price/v2/latest' in routes, 'v2 route missing'
print('all routes present')
"
```

Expected:
```
[..., '/predict/gas-price/latest', ..., '/predict/gas-price/v2/latest', ...]
all routes present
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: mount v2 LSTM router when CONFIG_V2 env var is set"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] `model/lstm.py` with `_LSTMNet` + `LSTMWrapper` — Task 3
- [x] Sliding window sequences in `fit()` — Task 3 Step 2
- [x] Val split last 10% + early stopping patience=5 — Task 3 Step 2
- [x] `predict()` returns `(len(X),)` with zero-padding — Task 3 Step 2
- [x] `joblib.dump/load` round-trip — tested in Task 2, implemented in Task 3
- [x] `configs/ethereum-gas-price-predictor-v2.yaml` — Task 4
- [x] `train.py` lstm branch — Task 5
- [x] `router_gas_price_v2.py` — Task 7
- [x] Auto-regression for `n_blocks > 1` — Task 7 Step 1
- [x] Response identical shape to v1 — Task 7 + tested in Task 6
- [x] `app.py` CONFIG_V2 with GCS support — Task 8
- [x] `tests/test_lstm.py` — Task 2
- [x] `tests/test_router_gas_price_v2.py` — Task 6

**Out-of-scope confirmed absent:**
- No POST `/predict/gas-price/v2` (single-row) — LSTM requires sequence context
- No HPO for LSTM
- No Streamlit UI changes — v2 response shape matches v1
