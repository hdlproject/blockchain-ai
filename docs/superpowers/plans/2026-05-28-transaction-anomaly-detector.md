# Transaction Anomaly Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DBSCAN-based transaction anomaly detector that sweeps recent Ethereum blocks for training data and exposes a `POST /detect/transaction` endpoint for pre-submission anomaly scoring.

**Architecture:** New `task: clustering` config type added alongside regression/classification. `DBSCANWrapper` (scaler + DBSCAN + NearestNeighbors) is trained on per-transaction features extracted from block sweeps, then served via `router_anomaly`. Streamlit gets a 5th tab with a raw-feature form connecting to a configurable backend URL.

**Tech Stack:** scikit-learn (DBSCAN, StandardScaler, NearestNeighbors), joblib, FastAPI, Streamlit, Etherscan API

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Modify | `src/blockchain_ai/config.py` | Accept `clustering` task; optional `test_path`; clustering serve/paths parsing |
| Create | `configs/transaction-anomaly-detector.yaml` | Clustering pipeline config |
| Create | `src/blockchain_ai/model/dbscan_wrapper.py` | DBSCANWrapper: fit/predict/anomaly_score |
| Create | `src/blockchain_ai/feature/transaction_features.py` | Per-transaction feature extraction + address window stats |
| Modify | `src/blockchain_ai/connector/etherscan.py` | Add `get_block_with_txs()` |
| Create | `src/blockchain_ai/workflow/collect_transactions.py` | Block sweep → transactions CSV → processed features CSV |
| Modify | `src/blockchain_ai/train.py` | Clustering branch (no split, fit on full data) |
| Modify | `src/blockchain_ai/evaluate.py` | Clustering branch (anomaly_ratio, n_clusters, n_noise) |
| Modify | `src/blockchain_ai/workflow/run_pipeline.py` | Clustering branch |
| Create | `src/blockchain_ai/server/router_anomaly.py` | POST /detect/transaction |
| Modify | `app.py` | Mount router_anomaly for task=clustering |
| Modify | `ui/streamlit_app.py` | 5th tab + Transaction Anomaly API URL + sidebar metrics |
| Create | `scripts/run_local_transaction_anomaly_detector.sh` | 3-step run script |

---

## Task 1: Config — clustering task support

**Files:**
- Modify: `src/blockchain_ai/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py — add these tests
import pytest
from blockchain_ai.config import load_config

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_config.py::test_load_clustering_config tests/test_config.py::test_clustering_config_paths_test_path_optional -v
```
Expected: FAIL — `ValueError: Config 'task' must be 'regression' or 'classification'`

- [ ] **Step 3: Update `PathsConfig` dataclass — make `test_path` optional**

In `src/blockchain_ai/config.py`, change `PathsConfig`:
```python
@dataclass
class PathsConfig:
    processed_path: str
    report_path: str
    test_path: str = ""
```

- [ ] **Step 4: Accept `clustering` in task validation**

Replace:
```python
    task = raw.get("task", "regression")
    if task not in ("regression", "classification"):
        raise ValueError(f"Config 'task' must be 'regression' or 'classification', got: {task!r}")
```
With:
```python
    task = raw.get("task", "regression")
    if task not in ("regression", "classification", "clustering"):
        raise ValueError(f"Config 'task' must be 'regression', 'classification', or 'clustering', got: {task!r}")
```

- [ ] **Step 5: Skip target_col/test_size validation for clustering**

Replace the ingest/train validation block:
```python
    for key in ("feature_cols", "fill_zero_cols", "target_col"):
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    for key in ("target_col", "model_type", "test_size", "hyperparameters"):
        if key not in t:
            raise ValueError(f"Config train section missing required key: '{key}'")
```
With:
```python
    required_ingest = ["feature_cols", "fill_zero_cols"] if task == "clustering" else ["feature_cols", "fill_zero_cols", "target_col"]
    for key in required_ingest:
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    required_train = ["model_type", "hyperparameters"] if task == "clustering" else ["target_col", "model_type", "test_size", "hyperparameters"]
    for key in required_train:
        if key not in t:
            raise ValueError(f"Config train section missing required key: '{key}'")
```

- [ ] **Step 6: Add clustering serve config parsing and fix paths parsing**

Replace the `if "serve" in raw:` block's else branch (classification) with a three-way branch, and fix paths to use optional `test_path`:

In the serve block, add clustering branch after the `else:` (classification) block:
```python
    serve_cfg = None
    if "serve" in raw:
        s = raw["serve"]
        if task == "regression":
            for key in ("title", "description", "model_path", "target_description", "target_unit", "fields"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                title=s["title"],
                description=s["description"].strip(),
                target_description=s["target_description"],
                target_unit=s["target_unit"],
                log_transform=bool(s.get("log_transform", False)),
                fields={
                    name: FieldConfig(
                        type=meta["type"],
                        description=meta["description"].strip(),
                        example=meta["example"],
                        ge=meta.get("ge"),
                        gt=meta.get("gt"),
                        le=meta.get("le"),
                        lt=meta.get("lt"),
                    )
                    for name, meta in s["fields"].items()
                },
            )
        elif task == "classification":
            for key in ("model_path", "confidence_threshold", "db_path"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                title=s.get("title"),
                description=s["description"].strip() if s.get("description") else None,
                confidence_threshold=float(s["confidence_threshold"]),
                db_path=s["db_path"],
            )
        else:  # clustering
            for key in ("model_path", "title", "description", "fields"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                title=s["title"],
                description=s["description"].strip(),
                fields={
                    name: FieldConfig(
                        type=meta["type"],
                        description=meta["description"].strip(),
                        example=meta["example"],
                        ge=meta.get("ge"),
                        gt=meta.get("gt"),
                        le=meta.get("le"),
                        lt=meta.get("lt"),
                    )
                    for name, meta in (s.get("fields") or {}).items()
                },
            )
```

Replace the paths block:
```python
    paths_cfg = None
    if "paths" in raw:
        p = raw["paths"]
        required_paths = ["processed_path", "report_path"]
        if task != "clustering":
            required_paths.append("test_path")
        for key in required_paths:
            if key not in p:
                raise ValueError(f"Config paths section missing required key: '{key}'")
        paths_cfg = PathsConfig(
            processed_path=p["processed_path"],
            test_path=p.get("test_path", ""),
            report_path=p["report_path"],
        )
```

Also remove the classification-only guard (it requires goplus/ofac sections — clustering should not require them):
```python
    if task == "classification":
        for section in ("goplus", "ofac"):
            if section not in raw:
                raise ValueError(f"Config task=classification requires section: '{section}'")
```
Leave this block unchanged — it only runs for classification.

Finally update the `TrainConfig` construction to handle missing keys for clustering:
```python
        train=TrainConfig(
            target_col=t.get("target_col", ""),
            model_type=t["model_type"],
            test_size=float(t.get("test_size", 0.0)),
            hyperparameters=t["hyperparameters"],
        ),
```
And `IngestConfig`:
```python
        ingest=IngestConfig(
            feature_cols=i["feature_cols"],
            fill_zero_cols=i["fill_zero_cols"],
            target_col=i.get("target_col", ""),
        ),
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
poetry run pytest tests/test_config.py -v
```
Expected: All pass including new clustering tests.

- [ ] **Step 8: Create `configs/transaction-anomaly-detector.yaml`**

```yaml
task: clustering

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

collect:
  n_blocks: 500
  output_path: data/raw/transactions.csv

ingest:
  feature_cols:
    - value_eth
    - gas_price_gwei
    - gas_used
    - input_data_len
    - is_contract_call
    - hour_of_day
    - sender_tx_count_window
    - sender_avg_value_eth
    - receiver_tx_count_window
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
  title: Ethereum Transaction Anomaly Detector
  description: Detects unusual Ethereum transactions using DBSCAN clustering.
  model_path: models/transaction_anomaly_detector.joblib
  fields:
    value_eth:
      type: float
      description: Transaction value in ETH
      example: 0.5
      ge: 0.0
    gas_price_gwei:
      type: float
      description: Gas price in Gwei
      example: 20.0
      ge: 0.0
    gas_used:
      type: float
      description: Gas consumed
      example: 21000.0
      ge: 0.0
    input_data_len:
      type: float
      description: Calldata byte length (0 = simple ETH transfer)
      example: 0.0
      ge: 0.0
    is_contract_call:
      type: float
      description: 1 if contract call, 0 if simple transfer
      example: 0.0
      ge: 0.0
      le: 1.0
    hour_of_day:
      type: float
      description: UTC hour of block timestamp (0-23)
      example: 14.0
      ge: 0.0
      le: 23.0
    sender_tx_count_window:
      type: float
      description: Sender transaction count in recent window
      example: 5.0
      ge: 1.0
    sender_avg_value_eth:
      type: float
      description: Sender average transaction value in ETH in recent window
      example: 0.5
      ge: 0.0
    receiver_tx_count_window:
      type: float
      description: Receiver transaction count in recent window
      example: 3.0
      ge: 1.0
```

- [ ] **Step 9: Commit**

```bash
git add src/blockchain_ai/config.py configs/transaction-anomaly-detector.yaml tests/test_config.py
git commit -m "feat: add clustering task type to config"
```

---

## Task 2: DBSCANWrapper

**Files:**
- Create: `src/blockchain_ai/model/dbscan_wrapper.py`
- Test: `tests/test_dbscan_wrapper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dbscan_wrapper.py
import numpy as np
import pytest
from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper


def _make_data():
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(80, 2))
    cluster_b = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(80, 2))
    outlier = np.array([[20.0, 20.0]])
    return np.vstack([cluster_a, cluster_b, outlier])


def test_fit_returns_self():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    result = model.fit(X)
    assert result is model


def test_predict_labels_outlier_as_anomaly():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    model.fit(X)
    outlier = np.array([[20.0, 20.0]])
    labels = model.predict(outlier)
    assert labels[0] == -1


def test_predict_labels_cluster_point_as_normal():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    model.fit(X)
    normal = np.array([[0.0, 0.0]])
    labels = model.predict(normal)
    assert labels[0] == 0


def test_anomaly_score_outlier_higher_than_normal():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    model.fit(X)
    normal_score = model.anomaly_score(np.array([[0.0, 0.0]]))[0]
    outlier_score = model.anomaly_score(np.array([[20.0, 20.0]]))[0]
    assert outlier_score > normal_score


def test_predict_before_fit_raises():
    model = DBSCANWrapper()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.array([[1.0, 2.0]]))


def test_all_noise_fallback():
    # When all points are noise, should still not raise
    X = np.eye(10) * 100  # 10 isolated points, no density
    model = DBSCANWrapper(eps=0.1, min_samples=5)
    model.fit(X)
    scores = model.anomaly_score(X)
    assert scores.shape == (10,)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_dbscan_wrapper.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'blockchain_ai.model.dbscan_wrapper'`

- [ ] **Step 3: Implement `DBSCANWrapper`**

Create `src/blockchain_ai/model/dbscan_wrapper.py`:
```python
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class DBSCANWrapper:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler: StandardScaler | None = None
        self._core_samples: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> "DBSCANWrapper":
        X = np.asarray(X, dtype=np.float32)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        db.fit(X_scaled)
        if len(db.core_sample_indices_) > 0:
            self._core_samples = X_scaled[db.core_sample_indices_]
        else:
            self._core_samples = X_scaled
        self._nn = NearestNeighbors(n_neighbors=1)
        self._nn.fit(self._core_samples)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None or self._scaler is None:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float32))
        distances, _ = self._nn.kneighbors(X_scaled)
        return distances[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        return np.where(scores <= self.eps, 0, -1).astype(np.int32)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_dbscan_wrapper.py -v
```
Expected: All 6 pass.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/model/dbscan_wrapper.py tests/test_dbscan_wrapper.py
git commit -m "feat: add DBSCANWrapper model"
```

---

## Task 3: Transaction feature extractor

**Files:**
- Create: `src/blockchain_ai/feature/transaction_features.py`
- Test: `tests/test_transaction_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_transaction_features.py
import pandas as pd
import pytest
from blockchain_ai.feature.transaction_features import TransactionFeatureExtractor


def _make_tx(frm="0xaaa", to="0xbbb", value_wei="1000000000000000000",
             gas="21000", gas_price="20000000000", input_data="0x",
             timestamp="1700000000"):
    return {
        "from": frm, "to": to, "value": value_wei,
        "gas": gas, "gasPrice": gas_price,
        "input": input_data, "timeStamp": timestamp,
    }


def test_extract_dataset_returns_dataframe():
    txs = [_make_tx() for _ in range(5)]
    extractor = TransactionFeatureExtractor()
    df = extractor.extract_dataset(txs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_extract_dataset_has_expected_columns():
    txs = [_make_tx()]
    extractor = TransactionFeatureExtractor()
    df = extractor.extract_dataset(txs)
    expected = {
        "value_eth", "gas_price_gwei", "gas_used", "input_data_len",
        "is_contract_call", "hour_of_day", "sender_tx_count_window",
        "sender_avg_value_eth", "receiver_tx_count_window",
    }
    assert expected.issubset(set(df.columns))


def test_value_eth_conversion():
    txs = [_make_tx(value_wei="2000000000000000000")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert abs(df["value_eth"].iloc[0] - 2.0) < 1e-6


def test_is_contract_call_with_calldata():
    txs = [_make_tx(input_data="0xabcdef")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["is_contract_call"].iloc[0] == 1.0


def test_is_contract_call_simple_transfer():
    txs = [_make_tx(input_data="0x")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["is_contract_call"].iloc[0] == 0.0


def test_sender_tx_count_window():
    txs = [_make_tx(frm="0xaaa") for _ in range(3)] + [_make_tx(frm="0xbbb")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    aaa_rows = df[df["sender_tx_count_window"] == 3.0]
    assert len(aaa_rows) == 3


def test_skips_tx_without_from():
    txs = [{"to": "0xbbb", "value": "0", "gas": "21000",
             "gasPrice": "1000000000", "input": "0x", "timeStamp": "1700000000"}]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert len(df) == 0


def test_input_data_len():
    # "0xabcd" = 2 bytes
    txs = [_make_tx(input_data="0xabcd")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["input_data_len"].iloc[0] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_transaction_features.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `TransactionFeatureExtractor`**

Create `src/blockchain_ai/feature/transaction_features.py`:
```python
from datetime import datetime, timezone
import pandas as pd


class TransactionFeatureExtractor:
    def extract_dataset(self, raw_txs: list[dict]) -> pd.DataFrame:
        sender_counts: dict[str, int] = {}
        sender_values: dict[str, list[float]] = {}
        receiver_counts: dict[str, int] = {}

        for tx in raw_txs:
            frm = tx.get("from", "").lower()
            to = tx.get("to", "").lower()
            value = int(tx.get("value", 0)) / 1e18
            if frm:
                sender_counts[frm] = sender_counts.get(frm, 0) + 1
                sender_values.setdefault(frm, []).append(value)
            if to:
                receiver_counts[to] = receiver_counts.get(to, 0) + 1

        sender_avg: dict[str, float] = {
            addr: sum(vals) / len(vals)
            for addr, vals in sender_values.items()
        }

        rows = [
            row for tx in raw_txs
            if (row := self._extract_one(tx, sender_counts, sender_avg, receiver_counts)) is not None
        ]
        return pd.DataFrame(rows)

    def _extract_one(
        self,
        tx: dict,
        sender_counts: dict[str, int],
        sender_avg: dict[str, float],
        receiver_counts: dict[str, int],
    ) -> dict | None:
        frm = tx.get("from", "").lower()
        to = tx.get("to", "").lower()
        if not frm:
            return None

        value_eth = int(tx.get("value", 0)) / 1e18
        gas_price_gwei = int(tx.get("gasPrice", 0)) / 1e9
        gas_used = float(int(tx.get("gas", 0)))
        input_data = tx.get("input", "0x") or "0x"
        input_data_len = float(max(0, (len(input_data) - 2) // 2)) if input_data != "0x" else 0.0
        is_contract_call = 1.0 if input_data != "0x" else 0.0
        timestamp = int(tx.get("timeStamp", tx.get("timestamp", 0)))
        hour_of_day = float(datetime.fromtimestamp(timestamp, tz=timezone.utc).hour) if timestamp else 0.0

        return {
            "value_eth": round(value_eth, 6),
            "gas_price_gwei": round(gas_price_gwei, 4),
            "gas_used": gas_used,
            "input_data_len": input_data_len,
            "is_contract_call": is_contract_call,
            "hour_of_day": hour_of_day,
            "sender_tx_count_window": float(sender_counts.get(frm, 1)),
            "sender_avg_value_eth": round(sender_avg.get(frm, value_eth), 6),
            "receiver_tx_count_window": float(receiver_counts.get(to, 1)),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_transaction_features.py -v
```
Expected: All 8 pass.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/feature/transaction_features.py tests/test_transaction_features.py
git commit -m "feat: add transaction feature extractor"
```

---

## Task 4: EtherscanClient — add `get_block_with_txs`

**Files:**
- Modify: `src/blockchain_ai/connector/etherscan.py`
- Test: `tests/test_etherscan.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_etherscan.py — add this test
from unittest.mock import patch, MagicMock
from blockchain_ai.connector.etherscan import EtherscanClient


def _make_client():
    with patch.dict("os.environ", {"ETHERSCAN_API_KEY": "testkey"}):
        return EtherscanClient("https://api.etherscan.io/v2/api", 1, 5, 30)


def test_get_block_with_txs_returns_list():
    client = _make_client()
    mock_result = {
        "timestamp": "0x65000000",
        "transactions": [
            {
                "from": "0xaaa",
                "to": "0xbbb",
                "value": "0xde0b6b3a7640000",
                "gas": "0x5208",
                "gasPrice": "0x4a817c800",
                "input": "0x",
            }
        ],
    }
    with patch.object(client, "_get", return_value=mock_result):
        rows = client.get_block_with_txs(12345)
    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["from"] == "0xaaa"
    assert rows[0]["value"] == "1000000000000000000"
    assert rows[0]["timeStamp"] == "1694287872"


def test_get_block_with_txs_returns_none_on_missing_result():
    client = _make_client()
    with patch.object(client, "_get", return_value=None):
        rows = client.get_block_with_txs(12345)
    assert rows is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_etherscan.py::test_get_block_with_txs_returns_list tests/test_etherscan.py::test_get_block_with_txs_returns_none_on_missing_result -v
```
Expected: FAIL — `AttributeError: 'EtherscanClient' object has no attribute 'get_block_with_txs'`

- [ ] **Step 3: Add `get_block_with_txs` to `EtherscanClient`**

Add at the end of `src/blockchain_ai/connector/etherscan.py`:
```python
    def get_block_with_txs(self, block_number: int) -> list[dict] | None:
        result = self._get({
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(block_number),
            "boolean": "true",
        })
        if result is None:
            return None
        timestamp = int(result.get("timestamp", "0x0"), 16)
        txs = result.get("transactions", [])
        if not isinstance(txs, list):
            return None
        rows = []
        for tx in txs:
            if not isinstance(tx, dict):
                continue
            rows.append({
                "from": tx.get("from", ""),
                "to": tx.get("to", "") or "",
                "value": str(int(tx.get("value", "0x0"), 16)),
                "gas": str(int(tx.get("gas", "0x0"), 16)),
                "gasPrice": str(int(tx.get("gasPrice", "0x0"), 16)),
                "input": tx.get("input", "0x"),
                "timeStamp": str(timestamp),
            })
        return rows
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_etherscan.py -v
```
Expected: All pass including new tests.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/connector/etherscan.py tests/test_etherscan.py
git commit -m "feat: add get_block_with_txs to EtherscanClient"
```

---

## Task 5: Transaction data collector

**Files:**
- Create: `src/blockchain_ai/workflow/collect_transactions.py`

- [ ] **Step 1: Create `collect_transactions.py`**

```python
#!/usr/bin/env python3
"""
Sweep recent Ethereum blocks for transactions and extract features for anomaly detection.

Usage:
    poetry run python src/blockchain_ai/workflow/collect_transactions.py \
        --config configs/transaction-anomaly-detector.yaml
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.transaction_features import TransactionFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Collect Ethereum transactions for anomaly detection")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.collect is None:
        raise RuntimeError("Config missing 'collect' section")
    if cfg.paths is None:
        raise RuntimeError("Config missing 'paths' section")

    client = EtherscanClient.from_config(cfg.etherscan)
    n_blocks = cfg.collect.n_blocks
    output_path = cfg.collect.output_path
    processed_path = cfg.paths.processed_path

    print(f"[1/3] Fetching latest block number ...")
    latest = client.get_latest_block_number()
    start_block = latest - n_blocks + 1
    print(f"      Fetching blocks {start_block} → {latest} ({n_blocks} blocks)")

    all_txs: list[dict] = []
    print(f"[2/3] Fetching transactions ...")
    for i, block_num in enumerate(range(start_block, latest + 1)):
        try:
            txs = client.get_block_with_txs(block_num)
            if txs:
                all_txs.extend(txs)
        except Exception as exc:
            warnings.warn(f"Block {block_num} failed: {exc}")
        if (i + 1) % 50 == 0:
            print(f"      Fetched {i + 1}/{n_blocks} blocks ({len(all_txs)} txs so far) ...")

    if not all_txs:
        raise RuntimeError("No transactions collected — check ETHERSCAN_API_KEY and network.")

    print(f"[3/3] Extracting features from {len(all_txs)} transactions ...")
    extractor = TransactionFeatureExtractor()
    df = extractor.extract_dataset(all_txs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Saved {len(df)} raw rows to {output_path}")

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df[cfg.ingest.feature_cols].to_csv(processed_path, index=False)
    print(f"      Saved {len(df)} processed rows to {processed_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports without error**

```bash
poetry run python -c "from blockchain_ai.workflow.collect_transactions import main; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/blockchain_ai/workflow/collect_transactions.py
git commit -m "feat: add collect_transactions workflow"
```

---

## Task 6: train.py — clustering branch

**Files:**
- Modify: `src/blockchain_ai/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_train.py — add this test
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from blockchain_ai.train import train_model
from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper


def test_train_clustering_returns_dbscan_wrapper(tmp_path):
    from blockchain_ai.config import (
        PipelineConfig, IngestConfig, TrainConfig, PathsConfig, ServeConfig
    )
    cfg = PipelineConfig(
        task="clustering",
        ingest=IngestConfig(feature_cols=["x", "y"], fill_zero_cols=[], target_col=""),
        train=TrainConfig(target_col="", model_type="dbscan", test_size=0.0,
                          hyperparameters={"eps": 0.5, "min_samples": 3}),
        paths=PathsConfig(processed_path=str(tmp_path / "p.csv"),
                          report_path=str(tmp_path / "r.json")),
        serve=ServeConfig(model_path=str(tmp_path / "model.joblib")),
    )
    df = pd.DataFrame({"x": np.random.randn(50), "y": np.random.randn(50)})
    df.to_csv(tmp_path / "p.csv", index=False)

    model = train_model(
        str(tmp_path / "p.csv"),
        str(tmp_path / "model.joblib"),
        "",
        cfg,
    )
    assert isinstance(model, DBSCANWrapper)
    assert (tmp_path / "model.joblib").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_train.py::test_train_clustering_returns_dbscan_wrapper -v
```
Expected: FAIL — `ValueError: Unknown task: 'clustering'`

- [ ] **Step 3: Add clustering branch to `train_model`**

In `src/blockchain_ai/train.py`, insert before the existing `df = pd.read_csv(input_path)` line, right after the HPO block:

```python
    if cfg.task == "clustering":
        if train_config.model_type == "dbscan":
            from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper
            model = DBSCANWrapper(**train_config.hyperparameters)
        else:
            raise ValueError(
                f"Unknown model_type: {train_config.model_type!r}. Supported: 'dbscan'"
            )
        df = pd.read_csv(input_path)
        X = df[cfg.ingest.feature_cols].values
        model.fit(X)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        return model
```

Also update the final `else: raise ValueError` to cover clustering (it's now handled above, but the existing else still fires for unknown tasks):
```python
    else:
        raise ValueError(f"Unknown task: {cfg.task!r}")
```
This remains unchanged — clustering is handled before reaching it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_train.py -v
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/train.py tests/test_train.py
git commit -m "feat: add clustering branch to train_model"
```

---

## Task 7: evaluate.py — clustering branch

**Files:**
- Modify: `src/blockchain_ai/evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_evaluate.py — add this test
import json
import numpy as np
import pandas as pd
import joblib
import pytest
from blockchain_ai.evaluate import evaluate_model
from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper


def test_evaluate_clustering(tmp_path):
    from blockchain_ai.config import (
        PipelineConfig, IngestConfig, TrainConfig, PathsConfig, ServeConfig
    )
    cfg = PipelineConfig(
        task="clustering",
        ingest=IngestConfig(feature_cols=["x", "y"], fill_zero_cols=[], target_col=""),
        train=TrainConfig(target_col="", model_type="dbscan", test_size=0.0,
                          hyperparameters={"eps": 0.5, "min_samples": 3}),
        paths=PathsConfig(processed_path=str(tmp_path / "p.csv"),
                          report_path=str(tmp_path / "r.json")),
        serve=ServeConfig(model_path=str(tmp_path / "model.joblib")),
    )
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2)).astype(np.float32)
    model = DBSCANWrapper(eps=0.5, min_samples=3)
    model.fit(X)
    joblib.dump(model, tmp_path / "model.joblib")

    df = pd.DataFrame(X, columns=["x", "y"])
    df.to_csv(tmp_path / "p.csv", index=False)

    report = evaluate_model(
        str(tmp_path / "p.csv"),
        str(tmp_path / "model.joblib"),
        str(tmp_path / "r.json"),
        cfg,
    )
    assert "anomaly_ratio" in report
    assert "n_clusters" in report
    assert "n_noise" in report
    assert 0.0 <= report["anomaly_ratio"] <= 1.0
    assert (tmp_path / "r.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_evaluate.py::test_evaluate_clustering -v
```
Expected: FAIL — `ValueError: Unsupported task type: 'clustering'`

- [ ] **Step 3: Add clustering branch to `evaluate.py`**

In `src/blockchain_ai/evaluate.py`, update `evaluate_model`:

```python
def evaluate_model(
        test_path: str,
        model_path: str,
        report_path: str,
        cfg: PipelineConfig,
) -> dict:
    if cfg.task == "clustering":
        import numpy as np
        df = pd.read_csv(test_path)
        X = df[cfg.ingest.feature_cols].values
        model = joblib.load(model_path)
        labels = model.predict(X)
        n_noise = int(np.sum(labels == -1))
        n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
        report = {
            "anomaly_ratio": round(float(n_noise / len(labels)), 4),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
        return report

    X, y = pre_evaluate_model(test_path, cfg)
    y_raw = joblib.load(model_path).predict(X)
    report = post_evaluate_model(y, y_raw, cfg)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_evaluate.py -v
```
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/evaluate.py tests/test_evaluate.py
git commit -m "feat: add clustering branch to evaluate_model"
```

---

## Task 8: run_pipeline.py — clustering branch

**Files:**
- Modify: `src/blockchain_ai/workflow/run_pipeline.py`

- [ ] **Step 1: Add clustering branch**

In `src/blockchain_ai/workflow/run_pipeline.py`, replace:

```python
    if cfg.task == "regression":
        ...
    elif cfg.task == "classification":
        ...
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")

    print(f"[2/3] Training {cfg.train.model_type} model ...")
    train_model(processed_path, model_path, test_path, cfg)

    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(test_path, model_path, report_path, cfg)

    print(f"\nPipeline complete. Report:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")
```

With:

```python
    if cfg.task == "regression":
        if not args.input:
            parser.error("--input is required for task=regression")

        from blockchain_ai.feature.feature_engineering import load_and_clean
        print(f"[1/3] Ingesting {args.input} ...")
        load_and_clean(args.input, processed_path, cfg)
    elif cfg.task == "classification":
        if args.input:
            processed_path = args.input

        if not Path(processed_path).exists():
            raise FileNotFoundError(
                f"{processed_path} not found. Run collect_address_features.py first."
            )
        print(f"[1/3] Ingesting {processed_path} ...")
    elif cfg.task == "clustering":
        if not Path(processed_path).exists():
            raise FileNotFoundError(
                f"{processed_path} not found. Run collect_transactions.py first."
            )
        print(f"[1/3] Using {processed_path} ...")
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")

    print(f"[2/3] Training {cfg.train.model_type} model ...")
    train_model(processed_path, model_path, test_path, cfg)

    eval_path = processed_path if cfg.task == "clustering" else test_path
    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(eval_path, model_path, report_path, cfg)

    print(f"\nPipeline complete. Report:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")
```

- [ ] **Step 2: Verify pipeline imports cleanly**

```bash
poetry run python -c "
import sys; sys.path.insert(0, 'src')
from blockchain_ai.config import load_config
cfg = load_config('configs/transaction-anomaly-detector.yaml')
print('task:', cfg.task, '| model_type:', cfg.train.model_type)
"
```
Expected: `task: clustering | model_type: dbscan`

- [ ] **Step 3: Commit**

```bash
git add src/blockchain_ai/workflow/run_pipeline.py
git commit -m "feat: add clustering branch to run_pipeline"
```

---

## Task 9: router_anomaly.py + app.py

**Files:**
- Create: `src/blockchain_ai/server/router_anomaly.py`
- Modify: `app.py`
- Test: `tests/test_router_anomaly.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_router_anomaly.py
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from blockchain_ai.server.router_anomaly import create_router
from blockchain_ai.config import ServeConfig, FieldConfig
from fastapi import FastAPI


def _make_app():
    model = MagicMock()
    model.eps = 0.5
    model.anomaly_score.return_value = np.array([0.8])  # > eps → anomaly

    serve = ServeConfig(
        model_path="models/test.joblib",
        title="Test",
        description="Test",
        fields={
            "value_eth": FieldConfig(type="float", description="value", example=0.5, ge=0.0),
        },
    )
    feature_cols = ["value_eth"]
    app = FastAPI()
    app.include_router(create_router(serve, feature_cols, model))
    return TestClient(app), model


def test_detect_transaction_anomaly():
    client, _ = _make_app()
    resp = client.post("/detect/transaction", json={"value_eth": 9999.0})
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly"] is True
    assert data["label"] == "anomaly"
    assert isinstance(data["score"], float)


def test_detect_transaction_normal():
    client, model = _make_app()
    model.anomaly_score.return_value = np.array([0.1])  # < eps → normal
    resp = client.post("/detect/transaction", json={"value_eth": 0.5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["anomaly"] is False
    assert data["label"] == "normal"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_router_anomaly.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create `router_anomaly.py`**

```python
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import Field, create_model

from blockchain_ai.config import FieldConfig, ServeConfig

_TYPE_MAP = {"float": float, "int": int}


def _pydantic_field(fc: FieldConfig) -> Any:
    constraints: dict[str, Any] = {"description": fc.description, "examples": [fc.example]}
    if fc.ge is not None:
        constraints["ge"] = fc.ge
    if fc.gt is not None:
        constraints["gt"] = fc.gt
    if fc.le is not None:
        constraints["le"] = fc.le
    if fc.lt is not None:
        constraints["lt"] = fc.lt
    return (_TYPE_MAP[fc.type], Field(**constraints))


def create_router(serve: ServeConfig, feature_cols: list[str], model) -> APIRouter:
    router = APIRouter()

    TxModel = create_model(
        "TransactionInput",
        **{name: _pydantic_field(fc) for name, fc in (serve.fields or {}).items()},
    )

    @router.post("/detect/transaction", summary="Detect anomalous transaction")
    def detect_transaction(tx: TxModel):  # type: ignore[valid-type]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available.")
        df = pd.DataFrame([tx.model_dump()])[feature_cols]
        score = float(model.anomaly_score(df.values)[0])
        anomaly = score > model.eps
        return {
            "anomaly": anomaly,
            "score": round(score, 4),
            "label": "anomaly" if anomaly else "normal",
        }

    return router
```

- [ ] **Step 4: Add clustering branch to `app.py`**

After the classification branch in `app.py`, add:
```python
elif model is not None and task == "clustering":
    from blockchain_ai.server.router_anomaly import create_router as create_anomaly_router
    app.include_router(create_anomaly_router(serve, feature_cols, model))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
poetry run pytest tests/test_router_anomaly.py -v
```
Expected: Both pass.

- [ ] **Step 6: Commit**

```bash
git add src/blockchain_ai/server/router_anomaly.py app.py tests/test_router_anomaly.py
git commit -m "feat: add anomaly detection router and mount in app"
```

---

## Task 10: Streamlit UI — 5th tab

**Files:**
- Modify: `ui/streamlit_app.py`

- [ ] **Step 1: Add constants and detect function**

At the top of `ui/streamlit_app.py`, add after the existing URL defaults:
```python
_DEFAULT_TXANOMALY_URL = os.environ.get("TXANOMALY_API_URL", "http://localhost:8000")
TXANOMALY_REPORT_PATH = Path(__file__).parent.parent / "reports" / "transaction_anomaly.json"
```

Add the API function after `classify_address`:
```python
def detect_transaction(api_url: str, payload: dict) -> dict:
    resp = requests.post(f"{api_url}/detect/transaction", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()
```

Add the result renderer:
```python
def _render_anomaly_result(result: dict) -> None:
    score = result.get("score", 0.0)
    if result.get("anomaly"):
        st.error(f"Anomaly detected")
    else:
        st.success(f"Normal transaction")
    st.metric("Anomaly Score", f"{score:.4f}", help="Distance to nearest normal cluster. Higher = more unusual.")
```

- [ ] **Step 2: Add URL to sidebar loop**

In `main()`, update the sidebar URL loop:
```python
        for key, label, default in [
            ("gas_v1_url", "Gas Price v1", _DEFAULT_GAS_V1_URL),
            ("gas_v2_url", "Gas Price v2 (LSTM)", _DEFAULT_GAS_V2_URL),
            ("addr_url", "Address Classifier", _DEFAULT_ADDR_URL),
            ("txanomaly_url", "Transaction Anomaly", _DEFAULT_TXANOMALY_URL),
        ]:
```

- [ ] **Step 3: Add Transaction Anomaly metrics to sidebar**

After the `addr_metrics` block in the sidebar, add:
```python
        txanomaly_metrics = load_metrics(TXANOMALY_REPORT_PATH)
        if txanomaly_metrics:
            st.caption("Transaction Anomaly")
            st.metric("Anomaly Ratio", f"{txanomaly_metrics['anomaly_ratio']:.4f}")
            st.metric("Clusters Found", str(txanomaly_metrics['n_clusters']))
            st.metric("Noise Points", str(txanomaly_metrics['n_noise']))
```

- [ ] **Step 4: Add 5th tab**

Update the tab declaration:
```python
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gas Price: Live", "Gas Price: Manual", "Gas Price v2 (LSTM)",
        "Address Classifier", "Transaction Anomaly",
    ])
```

Add `with tab5:` block after `with tab4:`:
```python
    with tab5:
        st.header("Transaction Anomaly Detector")
        st.write("Check whether a transaction looks unusual before submitting it.")

        col1, col2 = st.columns(2)
        with col1:
            value_eth = st.number_input("Value (ETH)", min_value=0.0, value=0.5, step=0.1,
                                         help="Transaction value in ETH")
            gas_price_gwei = st.number_input("Gas Price (Gwei)", min_value=0.0, value=20.0, step=1.0)
            gas_used = st.number_input("Gas Used", min_value=0.0, value=21000.0, step=1000.0)
            is_contract_call = st.selectbox("Contract Call?", options=[0.0, 1.0],
                                             format_func=lambda x: "Yes" if x == 1.0 else "No")
            input_data_len = st.number_input("Calldata Length (bytes)", min_value=0.0, value=0.0, step=4.0,
                                              help="0 = simple ETH transfer")
        with col2:
            hour_of_day = st.slider("Hour of Day (UTC)", min_value=0.0, max_value=23.0, value=14.0, step=1.0)
            sender_tx_count_window = st.number_input("Sender Tx Count (window)", min_value=1.0, value=5.0, step=1.0)
            sender_avg_value_eth = st.number_input("Sender Avg Value ETH (window)", min_value=0.0, value=0.5, step=0.1)
            receiver_tx_count_window = st.number_input("Receiver Tx Count (window)", min_value=1.0, value=3.0, step=1.0)

        if st.button("Check Transaction", key="btn_anomaly"):
            payload = {
                "value_eth": value_eth,
                "gas_price_gwei": gas_price_gwei,
                "gas_used": gas_used,
                "input_data_len": input_data_len,
                "is_contract_call": is_contract_call,
                "hour_of_day": hour_of_day,
                "sender_tx_count_window": sender_tx_count_window,
                "sender_avg_value_eth": sender_avg_value_eth,
                "receiver_tx_count_window": receiver_tx_count_window,
            }
            with st.spinner("Checking transaction..."):
                try:
                    result = detect_transaction(st.session_state.txanomaly_url, payload)
                    _render_anomaly_result(result)
                except requests.ConnectionError:
                    st.error("Could not connect to the Transaction Anomaly API. Is the backend running?")
                except requests.Timeout:
                    st.error("The request timed out. Please try again.")
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response else "?"
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    st.error(f"**HTTP {code}:** {detail}")
                except Exception as e:
                    with st.expander("Error details"):
                        st.exception(e)
```

- [ ] **Step 5: Verify syntax**

```bash
poetry run python -c "import ast; ast.parse(open('ui/streamlit_app.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add ui/streamlit_app.py
git commit -m "feat: add Transaction Anomaly tab to Streamlit UI"
```

---

## Task 11: Run script

**Files:**
- Create: `scripts/run_local_transaction_anomaly_detector.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# Run the full transaction anomaly detection pipeline locally.
# Collects transactions from recent blocks, trains DBSCAN, and starts the API + UI.
#
# Usage:
#   ./scripts/run_local_transaction_anomaly_detector.sh [CONFIG]
#
# Example:
#   ./scripts/run_local_transaction_anomaly_detector.sh configs/transaction-anomaly-detector.yaml
set -euo pipefail

CONFIG="${1:-configs/transaction-anomaly-detector.yaml}"

echo "==> Config : ${CONFIG}"
echo ""

echo "[1/3] Collecting transactions from Etherscan..."
poetry run python src/blockchain_ai/workflow/collect_transactions.py --config "${CONFIG}"

echo "[2/3] Running training pipeline..."
poetry run python src/blockchain_ai/workflow/run_pipeline.py \
  --config "${CONFIG}"

echo "[3/3] Starting API + Streamlit UI..."
echo "      FastAPI  → http://localhost:8000"
echo "      Streamlit → http://localhost:8501"
echo "      Press Ctrl+C to stop."
echo ""

trap 'kill ${API_PID} 2>/dev/null; exit 0' INT TERM

CONFIG="${CONFIG}" poetry run uvicorn app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

poetry run streamlit run ui/streamlit_app.py

kill "${API_PID}" 2>/dev/null
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/run_local_transaction_anomaly_detector.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_local_transaction_anomaly_detector.sh
git commit -m "feat: add transaction anomaly detector run script"
```

---

## Task 12: Full test suite

- [ ] **Step 1: Run all tests**

```bash
poetry run pytest --tb=short -q
```
Expected: All existing tests pass, no regressions.

- [ ] **Step 2: Smoke-test the config end-to-end**

```bash
poetry run python -c "
import sys; sys.path.insert(0, 'src')
from blockchain_ai.config import load_config
cfg = load_config('configs/transaction-anomaly-detector.yaml')
print('task:', cfg.task)
print('model_type:', cfg.train.model_type)
print('feature_cols:', cfg.ingest.feature_cols)
print('eps:', cfg.train.hyperparameters['eps'])
print('serve fields:', list(cfg.serve.fields.keys()))
"
```
Expected: prints all values cleanly with no errors.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete transaction anomaly detector implementation"
```
