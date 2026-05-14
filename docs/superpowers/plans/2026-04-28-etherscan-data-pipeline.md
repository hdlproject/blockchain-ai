# Etherscan Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the zip-based transaction dataset with Etherscan block-level data, retargeting the model to predict next-block `base_fee_per_gas` from network congestion features.

**Architecture:** A thin `EtherscanClient` handles all HTTP, a `collect_blocks.py` script fetches blocks and derives features, and the existing `ingest/train/serve` pipeline is updated to work with the new flat CSV and feature set. API key lives in `.env`, loaded via `python-dotenv`.

**Tech Stack:** Python 3.12, `requests`, `python-dotenv`, `pandas`, `xgboost`, `fastapi`, `poetry`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/blockchain_ai/etherscan.py` | HTTP wrapper for Etherscan API |
| Create | `scripts/collect_blocks.py` | Fetch blocks, derive features, write CSV |
| Create | `tests/test_etherscan.py` | Unit tests for EtherscanClient |
| Create | `tests/test_collect.py` | Unit tests for feature derivation |
| Create | `.env.example` | Template for API key |
| Modify | `src/blockchain_ai/config.py` | Add `EtherscanConfig`, `CollectConfig`; make `stratify_col` optional |
| Modify | `src/blockchain_ai/ingest.py` | Replace zip logic with plain CSV read |
| Modify | `src/blockchain_ai/train.py` | Make `stratify` conditional on `stratify_col` |
| Modify | `src/blockchain_ai/tune.py` | Make `stratify` conditional on `stratify_col` |
| Modify | `scripts/run_pipeline.py` | Rename `--raw` to `--input` |
| Modify | `tests/test_ingest.py` | Replace zip fixture with plain CSV |
| Modify | `tests/test_config.py` | Add tests for new config sections; make `stratify_col` optional |
| Modify | `tests/test_train.py` | Remove `stratify_col` from fixtures |
| Modify | `tests/test_tune.py` | Remove `stratify_col` from fixtures |
| Modify | `configs/ethereum-gas-price-predictor.yaml` | Add `etherscan`/`collect` sections; replace features; remove `stratify_col` |
| Modify | `app.py` | Add `load_dotenv()`; update `serve.fields` |
| Modify | `pyproject.toml` | Add `requests`, `python-dotenv` dependencies |
| Modify | `.gitignore` | Add `.env` |

---

## Task 1: Add dependencies and .env setup

**Files:**
- Modify: `pyproject.toml`
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Add dependencies via poetry**

```bash
poetry add requests python-dotenv
```

Expected output: poetry installs both packages and updates `poetry.lock`.

- [ ] **Step 2: Create `.env.example`**

```
ETHERSCAN_API_KEY=your_etherscan_api_key_here
```

Save to `.env.example`.

- [ ] **Step 3: Add `.env` to `.gitignore`**

Open `.gitignore` (create it if it doesn't exist) and add:

```
.env
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock .env.example .gitignore
git commit -m "chore: add requests, python-dotenv deps and .env setup"
```

---

## Task 2: Add `EtherscanConfig` and `CollectConfig` to config.py; make `stratify_col` optional

**Files:**
- Modify: `src/blockchain_ai/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new config sections**

Add to `tests/test_config.py`:

```python
from blockchain_ai.config import load_config, PipelineConfig, IngestConfig, TrainConfig, HpoConfig, ServeConfig, FieldConfig, EtherscanConfig, CollectConfig

_ETHERSCAN_YAML = """
etherscan:
  base_url: https://api.etherscan.io/api
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
    assert cfg.etherscan.base_url == "https://api.etherscan.io/api"
    assert cfg.etherscan.rate_limit_per_sec == 5
    assert cfg.etherscan.timeout_sec == 10


def test_load_config_with_collect_section(tmp_path):
    path = _write_yaml(tmp_path, _ETHERSCAN_YAML)
    cfg = load_config(path)
    assert cfg.collect is not None
    assert isinstance(cfg.collect, CollectConfig)
    assert cfg.collect.n_blocks == 500
    assert cfg.collect.output_path == "data/raw/blocks.csv"


def test_load_config_stratify_col_is_optional(tmp_path):
    path = _write_yaml(tmp_path, _ETHERSCAN_YAML)
    cfg = load_config(path)
    assert cfg.train.stratify_col is None


def test_load_config_without_etherscan_section(tmp_path):
    path = _write_yaml(tmp_path, _VALID_YAML)
    cfg = load_config(path)
    assert cfg.etherscan is None
    assert cfg.collect is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_config.py::test_load_config_with_etherscan_section tests/test_config.py::test_load_config_with_collect_section tests/test_config.py::test_load_config_stratify_col_is_optional tests/test_config.py::test_load_config_without_etherscan_section -v
```

Expected: FAIL — `EtherscanConfig`, `CollectConfig` not defined; `stratify_col` required.

- [ ] **Step 3: Update `src/blockchain_ai/config.py`**

Replace the entire file with:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class IngestConfig:
    feature_cols: list[str]
    fill_zero_cols: list[str]
    target_col: str


@dataclass
class TrainConfig:
    target_col: str
    model_type: str
    test_size: float
    hyperparameters: dict
    stratify_col: str | None = None


@dataclass
class HpoConfig:
    n_trials: int


@dataclass
class FieldConfig:
    type: str
    description: str
    example: Any
    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None


@dataclass
class ServeConfig:
    title: str
    description: str
    model_path: str
    target_description: str
    target_unit: str
    log_transform: bool
    fields: dict[str, FieldConfig]


@dataclass
class EtherscanConfig:
    base_url: str
    rate_limit_per_sec: int
    timeout_sec: int


@dataclass
class CollectConfig:
    n_blocks: int
    output_path: str


@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    hpo: "HpoConfig | None" = None
    serve: "ServeConfig | None" = None
    etherscan: "EtherscanConfig | None" = None
    collect: "CollectConfig | None" = None


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

    for key in ("feature_cols", "fill_zero_cols", "target_col"):
        if key not in i:
            raise ValueError(f"Config ingest section missing required key: '{key}'")

    for key in ("target_col", "model_type", "test_size", "hyperparameters"):
        if key not in t:
            raise ValueError(f"Config train section missing required key: '{key}'")

    hpo_cfg = None
    if "hpo" in raw:
        h = raw["hpo"]
        if not h or "n_trials" not in h:
            raise ValueError("Config hpo section missing required key: 'n_trials'")
        hpo_cfg = HpoConfig(n_trials=h["n_trials"])

    serve_cfg = None
    if "serve" in raw:
        s = raw["serve"]
        for key in ("title", "description", "model_path", "target_description", "target_unit", "fields"):
            if key not in s:
                raise ValueError(f"Config serve section missing required key: '{key}'")
        serve_cfg = ServeConfig(
            title=s["title"],
            description=s["description"].strip(),
            model_path=s["model_path"],
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

    etherscan_cfg = None
    if "etherscan" in raw:
        e = raw["etherscan"]
        for key in ("base_url", "rate_limit_per_sec", "timeout_sec"):
            if key not in e:
                raise ValueError(f"Config etherscan section missing required key: '{key}'")
        etherscan_cfg = EtherscanConfig(
            base_url=e["base_url"],
            rate_limit_per_sec=int(e["rate_limit_per_sec"]),
            timeout_sec=int(e["timeout_sec"]),
        )

    collect_cfg = None
    if "collect" in raw:
        c = raw["collect"]
        for key in ("n_blocks", "output_path"):
            if key not in c:
                raise ValueError(f"Config collect section missing required key: '{key}'")
        collect_cfg = CollectConfig(
            n_blocks=int(c["n_blocks"]),
            output_path=c["output_path"],
        )

    return PipelineConfig(
        ingest=IngestConfig(
            feature_cols=i["feature_cols"],
            fill_zero_cols=i["fill_zero_cols"],
            target_col=i["target_col"],
        ),
        train=TrainConfig(
            target_col=t["target_col"],
            model_type=t["model_type"],
            stratify_col=t.get("stratify_col"),
            test_size=t["test_size"],
            hyperparameters=t["hyperparameters"],
        ),
        hpo=hpo_cfg,
        serve=serve_cfg,
        etherscan=etherscan_cfg,
        collect=collect_cfg,
    )
```

- [ ] **Step 4: Fix existing `test_config.py` tests that still pass `stratify_col` as required**

In `tests/test_config.py`, update `_VALID_YAML` to keep `stratify_col` (backward compat — it's still optional not removed), and update the `test_load_config_train_fields` assertion to still check it when present:

```python
# _VALID_YAML already has stratify_col: transaction_type — leave it as-is.
# test_load_config_train_fields already asserts cfg.train.stratify_col == "transaction_type" — leave it.
# No changes needed to existing tests — stratify_col is optional but still parsed when present.
```

- [ ] **Step 5: Run all config tests**

```bash
poetry run pytest tests/test_config.py -v
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/blockchain_ai/config.py tests/test_config.py
git commit -m "feat: add EtherscanConfig, CollectConfig; make stratify_col optional"
```

---

## Task 3: Make `stratify_col` optional in `train.py` and `tune.py`

**Files:**
- Modify: `src/blockchain_ai/train.py`
- Modify: `src/blockchain_ai/tune.py`
- Modify: `tests/test_train.py`
- Modify: `tests/test_tune.py`

- [ ] **Step 1: Write failing test for train without stratify_col**

Add to `tests/test_train.py`:

```python
def _processed_df_no_stratify():
    return pd.DataFrame({
        "base_fee_gwei": [10.0 + i * 0.1 for i in range(20)],
        "gas_used_ratio": [0.5] * 20,
        "base_fee_trend": [0.01] * 20,
        "hour_of_day": [i % 24 for i in range(20)],
        "day_of_week": [i % 7 for i in range(20)],
        "log_base_fee_gwei": [2.3 + i * 0.01 for i in range(20)],
    })


def test_train_model_works_without_stratify_col(tmp_path):
    csv_path = tmp_path / "processed.csv"
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    _processed_df_no_stratify().to_csv(csv_path, index=False)

    cfg = TrainConfig(
        target_col="log_base_fee_gwei",
        model_type="xgboost",
        stratify_col=None,
        test_size=0.2,
        hyperparameters={"n_estimators": 10, "random_state": 42},
    )
    train_model(str(csv_path), str(model_path), str(test_path), cfg)

    assert model_path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
poetry run pytest tests/test_train.py::test_train_model_works_without_stratify_col -v
```

Expected: FAIL — `stratify=df[None]` raises `KeyError`.

- [ ] **Step 3: Update `src/blockchain_ai/train.py`**

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

    stratify = df[config.stratify_col] if config.stratify_col else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.hyperparameters.get("random_state", 42),
        stratify=stratify,
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

- [ ] **Step 4: Update `src/blockchain_ai/tune.py`**

Open `src/blockchain_ai/tune.py`. Find the `train_test_split` call and replace:

```python
# Before:
stratify=df[config.stratify_col],

# After:
stratify=df[config.stratify_col] if config.stratify_col else None,
```

- [ ] **Step 5: Update test fixtures to remove `stratify_col` from new-style configs**

In `tests/test_train.py`, update `_default_config` — keep existing `stratify_col="transaction_type"` for backward-compat tests, the new test uses `stratify_col=None` explicitly.

In `tests/test_tune.py`, update `_base_config`:

```python
def _base_config(**overrides):
    base = dict(
        target_col="log_gas_price",
        model_type="xgboost",
        stratify_col="transaction_type",   # keep — existing tests still pass transaction_type df
        test_size=0.2,
        hyperparameters={"n_estimators": 5, "random_state": 42},
    )
    base.update(overrides)
    return TrainConfig(**base)
```

No changes needed here — existing tests still work because `stratify_col` is optional with a default of `None`, and passing it explicitly still works.

- [ ] **Step 6: Run all train and tune tests**

```bash
poetry run pytest tests/test_train.py tests/test_tune.py -v
```

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/blockchain_ai/train.py src/blockchain_ai/tune.py tests/test_train.py
git commit -m "feat: make stratify_col optional in train and tune"
```

---

## Task 4: Build `EtherscanClient`

**Files:**
- Create: `src/blockchain_ai/etherscan.py`
- Create: `tests/test_etherscan.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_etherscan.py`:

```python
import os
import time
from unittest.mock import patch, MagicMock
import pytest
from blockchain_ai.etherscan import EtherscanClient


@pytest.fixture
def client():
    with patch.dict(os.environ, {"ETHERSCAN_API_KEY": "testkey"}):
        return EtherscanClient(
            base_url="https://api.etherscan.io/api",
            rate_limit_per_sec=100,  # high limit so tests don't sleep
            timeout_sec=10,
        )


def _ok_response(result):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "1", "message": "OK", "result": result}
    return m


def _error_response(message):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "0", "message": message, "result": None}
    return m


def test_client_reads_api_key_from_env():
    with patch.dict(os.environ, {"ETHERSCAN_API_KEY": "mykey123"}):
        c = EtherscanClient("https://api.etherscan.io/api", 5, 10)
        assert c._api_key == "mykey123"


def test_client_raises_if_api_key_missing():
    env = {k: v for k, v in os.environ.items() if k != "ETHERSCAN_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="ETHERSCAN_API_KEY"):
            EtherscanClient("https://api.etherscan.io/api", 5, 10)


def test_get_latest_block_number(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response("0xf4240")
        result = client.get_latest_block_number()
        assert result == 0xf4240
        assert mock_get.called


def test_get_latest_block_number_raises_on_http_error(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        m = MagicMock()
        m.status_code = 429
        mock_get.return_value = m
        with pytest.raises(RuntimeError, match="429"):
            client.get_latest_block_number()


def test_get_latest_block_number_raises_on_api_error(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("Invalid API Key")
        with pytest.raises(RuntimeError, match="Invalid API Key"):
            client.get_latest_block_number()


def test_get_fee_history_returns_list_of_dicts(client):
    fee_history_result = {
        "oldestBlock": "0xf4230",
        "baseFeePerGas": ["0x6fc23ac00", "0x6d54e9800", "0x0"],
        "gasUsedRatio": [0.95, 0.50],
    }
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(fee_history_result)
        result = client.get_fee_history(block_count=2, newest_block=0xf4240)
        # last entry in baseFeePerGas is the next block prediction — excluded
        assert len(result) == 2
        assert "base_fee_per_gas" in result[0]
        assert "gas_used_ratio" in result[0]
        assert "block_number" in result[0]


def test_get_fee_history_skips_pre_eip1559_blocks(client):
    fee_history_result = {
        "oldestBlock": "0xf4230",
        "baseFeePerGas": ["0x0", "0x6d54e9800", "0x0"],
        "gasUsedRatio": [0.95, 0.50],
    }
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(fee_history_result)
        result = client.get_fee_history(block_count=2, newest_block=0xf4240)
        # block with baseFeePerGas == 0x0 (first entry) is pre-EIP-1559 — skipped
        assert all(r["base_fee_per_gas"] > 0 for r in result)


def test_rate_limiting_sleeps_between_calls(client):
    slow_client = EtherscanClient.__new__(EtherscanClient)
    slow_client._api_key = "testkey"
    slow_client._base_url = "https://api.etherscan.io/api"
    slow_client._sleep_secs = 0.2
    slow_client._timeout = 10

    with patch("blockchain_ai.etherscan.requests.get") as mock_get, \
         patch("blockchain_ai.etherscan.time.sleep") as mock_sleep:
        mock_get.return_value = _ok_response("0xf4240")
        slow_client.get_latest_block_number()
        mock_sleep.assert_called_once_with(0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_etherscan.py -v
```

Expected: FAIL — `blockchain_ai.etherscan` module not found.

- [ ] **Step 3: Create `src/blockchain_ai/etherscan.py`**

```python
import os
import time
import requests
from blockchain_ai.config import EtherscanConfig


class EtherscanClient:
    def __init__(self, base_url: str, rate_limit_per_sec: int, timeout_sec: int):
        api_key = os.environ.get("ETHERSCAN_API_KEY")
        if not api_key:
            raise RuntimeError("ETHERSCAN_API_KEY environment variable is not set")
        self._api_key = api_key
        self._base_url = base_url
        self._sleep_secs = 1.0 / rate_limit_per_sec
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: EtherscanConfig) -> "EtherscanClient":
        return cls(
            base_url=config.base_url,
            rate_limit_per_sec=config.rate_limit_per_sec,
            timeout_sec=config.timeout_sec,
        )

    def _get(self, params: dict) -> dict:
        time.sleep(self._sleep_secs)
        params["apikey"] = self._api_key
        response = requests.get(self._base_url, params=params, timeout=self._timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Etherscan HTTP error: {response.status_code}")
        data = response.json()
        if str(data.get("status")) == "0":
            raise RuntimeError(f"Etherscan API error: {data.get('message')}")
        return data["result"]

    def get_latest_block_number(self) -> int:
        result = self._get({"module": "proxy", "action": "eth_blockNumber"})
        return int(result, 16)

    def get_fee_history(self, block_count: int, newest_block: int) -> list[dict]:
        result = self._get({
            "module": "proxy",
            "action": "eth_feeHistory",
            "blockCount": hex(block_count),
            "newestBlock": hex(newest_block),
            "rewardPercentiles": "",
        })
        oldest = int(result["oldestBlock"], 16)
        base_fees = result["baseFeePerGas"]
        ratios = result["gasUsedRatio"]

        # baseFeePerGas has block_count+1 entries (last is next-block prediction) — zip with ratios
        rows = []
        for i, (fee_hex, ratio) in enumerate(zip(base_fees, ratios)):
            base_fee = int(fee_hex, 16)
            if base_fee == 0:
                continue  # pre-EIP-1559 block
            rows.append({
                "block_number": oldest + i,
                "base_fee_per_gas": base_fee,
                "gas_used_ratio": float(ratio),
            })
        return rows
```

- [ ] **Step 4: Run tests**

```bash
poetry run pytest tests/test_etherscan.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/etherscan.py tests/test_etherscan.py
git commit -m "feat: add EtherscanClient"
```

---

## Task 5: Build feature derivation and `collect_blocks.py`

**Files:**
- Create: `scripts/collect_blocks.py`
- Create: `tests/test_collect.py`

- [ ] **Step 1: Write failing tests for feature derivation**

Create `tests/test_collect.py`:

```python
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_blocks import derive_features, apply_target_shift


def _raw_rows(n=15):
    return [
        {
            "block_number": 1000 + i,
            "base_fee_per_gas": (10 + i) * 10**9,  # 10–24 Gwei in Wei
            "gas_used_ratio": 0.5 + (i % 5) * 0.1,
            # unix timestamp: 2024-01-01 00:00:00 UTC + 12s per block
            "timestamp": 1704067200 + i * 12,
        }
        for i in range(n)
    ]


def test_derive_features_adds_base_fee_gwei(tmp_path):
    rows = _raw_rows()
    df = derive_features(rows)
    assert "base_fee_gwei" in df.columns
    assert abs(df["base_fee_gwei"].iloc[0] - 10.0) < 1e-6


def test_derive_features_preserves_gas_used_ratio(tmp_path):
    rows = _raw_rows()
    df = derive_features(rows)
    assert "gas_used_ratio" in df.columns
    assert df["gas_used_ratio"].iloc[0] == pytest.approx(0.5)


def test_derive_features_adds_hour_of_day(tmp_path):
    rows = _raw_rows()
    df = derive_features(rows)
    assert "hour_of_day" in df.columns
    assert 0 <= df["hour_of_day"].iloc[0] <= 23


def test_derive_features_adds_day_of_week(tmp_path):
    rows = _raw_rows()
    df = derive_features(rows)
    assert "day_of_week" in df.columns
    assert 0 <= df["day_of_week"].iloc[0] <= 6


def test_derive_features_adds_base_fee_trend(tmp_path):
    rows = _raw_rows(15)
    df = derive_features(rows)
    assert "base_fee_trend" in df.columns
    # first 10 rows have no 10-block lookback — filled with 0.0
    assert df["base_fee_trend"].iloc[0] == pytest.approx(0.0)
    # row 10 has a valid trend
    assert df["base_fee_trend"].iloc[10] != pytest.approx(0.0)


def test_apply_target_shift_adds_target_column():
    df = pd.DataFrame({"base_fee_gwei": [10.0, 11.0, 12.0, 13.0]})
    result = apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")
    assert "base_fee_gwei" in result.columns
    # target[0] = base_fee_gwei[1] = 11.0
    assert result["base_fee_gwei"].iloc[0] == pytest.approx(11.0)


def test_apply_target_shift_drops_last_row():
    df = pd.DataFrame({"base_fee_gwei": [10.0, 11.0, 12.0, 13.0]})
    result = apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")
    assert len(result) == 3


def test_apply_target_shift_raises_if_too_few_rows():
    df = pd.DataFrame({"base_fee_gwei": [10.0]})
    with pytest.raises(RuntimeError, match="too few"):
        apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_collect.py -v
```

Expected: FAIL — `collect_blocks` module not found.

- [ ] **Step 3: Create `scripts/collect_blocks.py`**

```python
#!/usr/bin/env python3
"""
Fetch the latest N blocks from Etherscan and write a feature CSV for training.

Usage:
    poetry run python scripts/collect_blocks.py --config configs/ethereum-gas-price-predictor.yaml
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.etherscan import EtherscanClient


def derive_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["base_fee_gwei"] = df["base_fee_per_gas"] / 1e9
    df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.dayofweek
    lookback = 10
    shifted = df["base_fee_gwei"].shift(lookback)
    df["base_fee_trend"] = ((df["base_fee_gwei"] - shifted) / shifted).fillna(0.0)
    return df


def apply_target_shift(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    if len(df) < 2:
        raise RuntimeError(f"Dataset has too few rows ({len(df)}) to apply target shift")
    df = df.copy()
    df[target_col] = df[source_col].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect Ethereum blocks from Etherscan")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.collect is None:
        raise RuntimeError("Config missing 'collect' section")

    client = EtherscanClient.from_config(cfg.etherscan)
    n_blocks = cfg.collect.n_blocks
    output_path = cfg.collect.output_path

    print(f"[1/3] Fetching latest block number ...")
    latest = client.get_latest_block_number()
    print(f"      Latest block: {latest}")

    print(f"[2/3] Fetching fee history for {n_blocks} blocks ...")
    rows: list[dict] = []
    remaining = n_blocks
    newest = latest

    while remaining > 0:
        batch = min(remaining, 1024)
        batch_rows = client.get_fee_history(block_count=batch, newest_block=newest)
        if not batch_rows:
            warnings.warn(f"No rows returned for newest_block={newest}")
            break
        rows = batch_rows + rows
        newest = batch_rows[0]["block_number"] - 1
        remaining -= batch
        print(f"      Fetched {len(rows)} blocks so far ...")

    if len(rows) < n_blocks:
        warnings.warn(f"Requested {n_blocks} blocks but only got {len(rows)}")

    print(f"[3/3] Deriving features and writing to {output_path} ...")
    df = derive_features(rows)
    df = apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
poetry run pytest tests/test_collect.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/collect_blocks.py tests/test_collect.py
git commit -m "feat: add collect_blocks script with feature derivation"
```

---

## Task 6: Replace zip-based `ingest.py` with plain CSV

**Files:**
- Modify: `src/blockchain_ai/ingest.py`
- Modify: `tests/test_ingest.py`

- [ ] **Step 1: Update `tests/test_ingest.py` — replace zip fixture with plain CSV**

Replace the entire file:

```python
import numpy as np
import pandas as pd
import pytest
from blockchain_ai.config import IngestConfig
from blockchain_ai.ingest import load_and_clean


def _minimal_df(**overrides):
    base = {
        "base_fee_gwei": [15.0],
        "gas_used_ratio": [0.75],
        "base_fee_trend": [0.02],
        "hour_of_day": [14],
        "day_of_week": [1],
        "extra_col": ["drop_me"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _default_config():
    return IngestConfig(
        feature_cols=["base_fee_gwei", "gas_used_ratio", "base_fee_trend", "hour_of_day", "day_of_week"],
        fill_zero_cols=[],
        target_col="base_fee_gwei",
    )


def test_load_and_clean_keeps_only_feature_cols(tmp_path):
    csv_path = str(tmp_path / "blocks.csv")
    _minimal_df().to_csv(csv_path, index=False)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, _default_config())

    assert "extra_col" not in result.columns
    assert "base_fee_gwei" not in result.columns  # target col is dropped after log transform
    assert "log_base_fee_gwei" in result.columns


def test_load_and_clean_adds_log_target(tmp_path):
    csv_path = str(tmp_path / "blocks.csv")
    _minimal_df(base_fee_gwei=[15.0]).to_csv(csv_path, index=False)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, _default_config())

    assert "log_base_fee_gwei" in result.columns
    expected = np.log1p(15.0)
    assert abs(result["log_base_fee_gwei"].iloc[0] - expected) < 1e-6


def test_load_and_clean_saves_csv(tmp_path):
    csv_path = str(tmp_path / "blocks.csv")
    _minimal_df().to_csv(csv_path, index=False)
    out_path = str(tmp_path / "out.csv")

    load_and_clean(csv_path, out_path, _default_config())

    assert (tmp_path / "out.csv").exists()


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.csv"), str(tmp_path / "out.csv"), _default_config())


def test_load_and_clean_fills_zero_cols(tmp_path):
    cfg = IngestConfig(
        feature_cols=["base_fee_gwei", "gas_used_ratio"],
        fill_zero_cols=["gas_used_ratio"],
        target_col="base_fee_gwei",
    )
    df = pd.DataFrame({"base_fee_gwei": [15.0], "gas_used_ratio": [None]})
    csv_path = str(tmp_path / "blocks.csv")
    df.to_csv(csv_path, index=False)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(csv_path, out_path, cfg)

    assert result["gas_used_ratio"].iloc[0] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: FAIL — `ingest.py` still reads a zip file, will fail on plain CSV path.

- [ ] **Step 3: Update `src/blockchain_ai/ingest.py`**

```python
import numpy as np
import pandas as pd
from pathlib import Path
from blockchain_ai.config import IngestConfig


def load_and_clean(input_path: str, output_path: str, config: IngestConfig) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(path)
    df = df[config.feature_cols + [config.target_col]]
    df[config.fill_zero_cols] = df[config.fill_zero_cols].fillna(0.0)
    df[f"log_{config.target_col}"] = np.log1p(df[config.target_col])
    df = df.drop(columns=[config.target_col])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
```

- [ ] **Step 4: Run tests**

```bash
poetry run pytest tests/test_ingest.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/ingest.py tests/test_ingest.py
git commit -m "refactor: replace zip-based ingest with plain CSV"
```

---

## Task 7: Update `run_pipeline.py` and YAML config

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `configs/ethereum-gas-price-predictor.yaml`

- [ ] **Step 1: Update `scripts/run_pipeline.py` — rename `--raw` to `--input`**

```python
#!/usr/bin/env python3
"""
Run the full regression pipeline: ingest -> [hpo] -> train -> evaluate.

Usage:
    poetry run python scripts/run_pipeline.py \
        --input data/raw/blocks.csv \
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
    parser.add_argument("--input", required=True, help="Path to raw input CSV")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_path = "data/processed/blocks.csv"
    test_path = "data/processed/blocks-test.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/4] Ingesting {args.input} ...")
    load_and_clean(args.input, processed_path, cfg.ingest)

    train_config = cfg.train
    if cfg.hpo is not None:
        print(f"[2/4] Running Optuna HPO ({cfg.hpo.n_trials} trials) ...")
        train_config = run_hpo(processed_path, cfg.train, n_trials=cfg.hpo.n_trials)
        print(f"      Best hyperparameters: {train_config.hyperparameters}")
    else:
        print(f"[2/4] Skipping HPO (no hpo section in config)")

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

- [ ] **Step 2: Replace `configs/ethereum-gas-price-predictor.yaml`**

```yaml
etherscan:
  base_url: https://api.etherscan.io/api
  rate_limit_per_sec: 5
  timeout_sec: 10

collect:
  n_blocks: 2000
  output_path: data/raw/blocks.csv

ingest:
  feature_cols:
    - base_fee_gwei
    - gas_used_ratio
    - base_fee_trend
    - hour_of_day
    - day_of_week
  fill_zero_cols: []
  target_col: base_fee_gwei

train:
  target_col: log_base_fee_gwei
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

hpo:
  n_trials: 50

serve:
  title: Ethereum Gas Price Predictor
  description: >
    Predicts the next block's base fee (EIP-1559 floor price) in Gwei
    given current network congestion signals.
  model_path: models/model.joblib
  target_description: Next block base fee
  target_unit: Gwei
  log_transform: true
  fields:
    base_fee_gwei:
      type: float
      description: Current block's base fee in Gwei.
      example: 15.0
      gt: 0
    gas_used_ratio:
      type: float
      description: >
        Current block gas used divided by gas limit (0.0 to 1.0).
        Values above 0.5 indicate congestion; above 0.9 is heavily congested.
      example: 0.75
      ge: 0
      le: 1
    base_fee_trend:
      type: float
      description: >
        Relative change in base fee over the last 10 blocks:
        (current - 10_blocks_ago) / 10_blocks_ago. Negative means fees are falling.
      example: 0.02
    hour_of_day:
      type: int
      description: UTC hour of the current block timestamp (0–23).
      example: 14
      ge: 0
      le: 23
    day_of_week:
      type: int
      description: UTC day of week of the current block timestamp (0=Monday, 6=Sunday).
      example: 1
      ge: 0
      le: 6
```

- [ ] **Step 3: Run the full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py configs/ethereum-gas-price-predictor.yaml
git commit -m "feat: update pipeline and config for Etherscan block features"
```

---

## Task 8: Update `app.py` with `load_dotenv` and new serve fields

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `load_dotenv()` to `app.py`**

Open `app.py`. Add at the top, before any other imports that might read env:

```python
from dotenv import load_dotenv
load_dotenv()
```

The full import block becomes:

```python
#!/usr/bin/env python3
import io
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field, create_model

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from blockchain_ai.config import FieldConfig, ServeConfig, load_config
```

No other changes to `app.py` are needed — it is already fully config-driven and will pick up the new `serve.fields` from the updated YAML automatically.

- [ ] **Step 2: Verify app loads cleanly**

```bash
poetry run python -c "import app; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add load_dotenv to app startup"
```

---

## Task 9: Final verification

- [ ] **Step 1: Run full test suite**

```bash
poetry run pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Verify collect script CLI help**

```bash
poetry run python scripts/collect_blocks.py --help
```

Expected: Prints usage with `--config` argument.

- [ ] **Step 3: Verify pipeline script CLI help**

```bash
poetry run python scripts/run_pipeline.py --help
```

Expected: Prints usage with `--input` and `--config` arguments.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: final cleanup and verification of Etherscan data pipeline"
```
