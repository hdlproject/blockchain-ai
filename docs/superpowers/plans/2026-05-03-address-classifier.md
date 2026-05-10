# Address Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an async REST API that classifies Ethereum addresses as `sanctioned`, `scammer`, `phishing`, or `unknown` using an XGBoost model trained on GoPlus/OFAC/Forta labels and Etherscan on-chain features.

**Architecture:** Unified YAML config drives label collection (GoPlus + OFAC + Forta), feature engineering (Etherscan), model training (XGBoost multi-class), and serving (FastAPI async job API with SQLite job store). Address is the idempotency key; first GET enqueues a background job, subsequent GETs poll the result.

**Tech Stack:** Python 3.12, XGBoost, FastAPI, SQLite (stdlib), requests, pytest

> **Note:** This plan supersedes `docs/superpowers/plans/2026-05-02-label-collection-pipeline.md`. Do not execute that plan.

---

## File Map

| Action | Path |
|--------|------|
| Modify | `src/blockchain_ai/config.py` |
| Create | `src/blockchain_ai/labels/__init__.py` |
| Create | `src/blockchain_ai/labels/schema.py` |
| Create | `src/blockchain_ai/labels/goplus.py` |
| Create | `src/blockchain_ai/labels/ofac.py` |
| Create | `src/blockchain_ai/labels/forta.py` |
| Create | `src/blockchain_ai/labels/unify.py` |
| Create | `configs/address-classifier.yaml` |
| Modify | `src/blockchain_ai/etherscan.py` |
| Create | `src/blockchain_ai/address_features.py` |
| Create | `src/blockchain_ai/job_store.py` |
| Modify | `src/blockchain_ai/train.py` |
| Modify | `src/blockchain_ai/evaluate.py` |
| Modify | `src/blockchain_ai/predict.py` |
| Create | `src/blockchain_ai/router_address.py` |
| Modify | `app.py` |
| Create | `scripts/collect_labels.py` |
| Create | `scripts/collect_address_features.py` |
| Modify | `scripts/run_pipeline.py` |
| Create | `tests/test_labels_schema.py` |
| Create | `tests/test_labels_goplus.py` |
| Create | `tests/test_labels_ofac.py` |
| Create | `tests/test_labels_forta.py` |
| Create | `tests/test_labels_unify.py` |
| Create | `tests/test_address_features.py` |
| Create | `tests/test_job_store.py` |
| Create | `tests/test_router_address.py` |
| Modify | `tests/test_config.py` |
| Modify | `tests/test_train.py` |
| Modify | `tests/test_evaluate.py` |
| Modify | `tests/test_predict.py` |

---

## Task 1: Config Extensions

**Files:**
- Modify: `src/blockchain_ai/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
_CLASSIFICATION_YAML = """
task: classification

goplus:
  base_url: https://api.gopluslabs.io/api/v1
  chain_id: 1
  rate_limit_per_sec: 2
  timeout_sec: 30

ofac:
  alt_url: https://www.treasury.gov/ofac/downloads/alt.csv
  timeout_sec: 60

forta:
  graphql_url: https://api.forta.network/graphql
  timeout_sec: 30
  max_alerts: 500
  scam_bot_ids:
    - "0xabc"

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
    assert cfg.forta is not None
    assert cfg.forta.max_alerts == 500
    assert cfg.forta.scam_bot_ids == ["0xabc"]
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_config.py::test_load_classification_config tests/test_config.py::test_regression_config_defaults_task_to_regression -v
```

Expected: FAIL with `TypeError` or `AttributeError` (task field not on PipelineConfig yet)

- [ ] **Step 3: Add new dataclasses and update PipelineConfig in `config.py`**

After the existing `CollectConfig` dataclass (line 63), insert:

```python
@dataclass
class GoPlusConfig:
    base_url: str
    chain_id: int
    rate_limit_per_sec: int
    timeout_sec: int


@dataclass
class OFACConfig:
    alt_url: str
    timeout_sec: int


@dataclass
class FortaConfig:
    graphql_url: str
    timeout_sec: int
    max_alerts: int
    scam_bot_ids: list[str]
```

- [ ] **Step 4: Update ServeConfig to make regression fields optional and add classification fields**

Replace the existing `ServeConfig` dataclass:

```python
@dataclass
class ServeConfig:
    model_path: str
    # regression-only
    title: str | None = None
    description: str | None = None
    target_description: str | None = None
    target_unit: str | None = None
    log_transform: bool = False
    fields: dict[str, FieldConfig] | None = None
    # classification-only
    confidence_threshold: float | None = None
    db_path: str | None = None
```

- [ ] **Step 5: Update PipelineConfig to add task and label source fields**

Replace the existing `PipelineConfig` dataclass:

```python
@dataclass
class PipelineConfig:
    ingest: IngestConfig
    train: TrainConfig
    task: str = "regression"
    hpo: "HpoConfig | None" = None
    serve: "ServeConfig | None" = None
    etherscan: "EtherscanConfig | None" = None
    collect: "CollectConfig | None" = None
    goplus: "GoPlusConfig | None" = None
    ofac: "OFACConfig | None" = None
    forta: "FortaConfig | None" = None
```

- [ ] **Step 6: Update `load_config()` to parse new sections and validate by task**

Replace the entire `load_config()` function:

```python
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

    task = raw.get("task", "regression")
    if task not in ("regression", "classification"):
        raise ValueError(f"Config 'task' must be 'regression' or 'classification', got: {task!r}")

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
        if "model_path" not in s:
            raise ValueError("Config serve section missing required key: 'model_path'")
        if task == "regression":
            for key in ("title", "description", "target_description", "target_unit", "fields"):
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
        else:
            for key in ("confidence_threshold", "db_path"):
                if key not in s:
                    raise ValueError(f"Config serve section missing required key: '{key}'")
            serve_cfg = ServeConfig(
                model_path=s["model_path"],
                confidence_threshold=float(s["confidence_threshold"]),
                db_path=s["db_path"],
            )

    etherscan_cfg = None
    if "etherscan" in raw:
        e = raw["etherscan"]
        for key in ("base_url", "chain_id", "rate_limit_per_sec", "timeout_sec"):
            if key not in e:
                raise ValueError(f"Config etherscan section missing required key: '{key}'")
        etherscan_cfg = EtherscanConfig(
            base_url=e["base_url"],
            chain_id=int(e["chain_id"]),
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
            checkpoint_every=int(c.get("checkpoint_every", 100)),
            max_history_blocks=int(c.get("max_history_blocks", 50_000)),
        )

    if task == "classification":
        for section in ("goplus", "ofac", "forta"):
            if section not in raw:
                raise ValueError(f"Config task=classification requires section: '{section}'")

    goplus_cfg = None
    if "goplus" in raw:
        g = raw["goplus"]
        for key in ("base_url", "chain_id", "rate_limit_per_sec", "timeout_sec"):
            if key not in g:
                raise ValueError(f"Config goplus section missing required key: '{key}'")
        goplus_cfg = GoPlusConfig(
            base_url=g["base_url"],
            chain_id=int(g["chain_id"]),
            rate_limit_per_sec=int(g["rate_limit_per_sec"]),
            timeout_sec=int(g["timeout_sec"]),
        )

    ofac_cfg = None
    if "ofac" in raw:
        o = raw["ofac"]
        for key in ("alt_url", "timeout_sec"):
            if key not in o:
                raise ValueError(f"Config ofac section missing required key: '{key}'")
        ofac_cfg = OFACConfig(
            alt_url=o["alt_url"],
            timeout_sec=int(o["timeout_sec"]),
        )

    forta_cfg = None
    if "forta" in raw:
        fo = raw["forta"]
        for key in ("graphql_url", "timeout_sec", "max_alerts", "scam_bot_ids"):
            if key not in fo:
                raise ValueError(f"Config forta section missing required key: '{key}'")
        forta_cfg = FortaConfig(
            graphql_url=fo["graphql_url"],
            timeout_sec=int(fo["timeout_sec"]),
            max_alerts=int(fo["max_alerts"]),
            scam_bot_ids=fo["scam_bot_ids"],
        )

    return PipelineConfig(
        task=task,
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
        goplus=goplus_cfg,
        ofac=ofac_cfg,
        forta=forta_cfg,
    )
```

- [ ] **Step 7: Run all config tests**

```
pytest tests/test_config.py -v
```

Expected: all tests PASS (existing + new)

- [ ] **Step 8: Commit**

```bash
git add src/blockchain_ai/config.py tests/test_config.py
git commit -m "feat: extend PipelineConfig for classification task with goplus/ofac/forta support"
```

---

## Task 2: Label Schema

**Files:**
- Create: `src/blockchain_ai/labels/__init__.py`
- Create: `src/blockchain_ai/labels/schema.py`
- Create: `tests/test_labels_schema.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_schema.py
import csv
from blockchain_ai.labels.schema import (
    AddressRecord, TokenRecord,
    write_address_csv, write_token_csv,
    ADDRESS_FIELDNAMES, TOKEN_FIELDNAMES,
)


def test_address_record_to_row_pipes_sources_and_flags():
    r = AddressRecord(
        address="0xabc", chain_id=1, label="sanctioned", confidence=1.0,
        sources=["ofac", "forta"], flags=["ofac_sdn", "scam_alert"],
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    row = r.to_row()
    assert row["sources"] == "ofac|forta"
    assert row["flags"] == "ofac_sdn|scam_alert"


def test_address_record_empty_lists_produce_empty_strings():
    r = AddressRecord("0xabc", 1, "unknown", 0.0, [], [], "2026-01-01T00:00:00+00:00")
    row = r.to_row()
    assert row["sources"] == ""
    assert row["flags"] == ""


def test_token_record_to_row():
    r = TokenRecord(
        token_address="0x123", chain_id=1, is_risky=True, risk_score=0.75,
        sources=["goplus"], flags=["honeypot"], fetched_at="2026-01-01T00:00:00+00:00",
    )
    row = r.to_row()
    assert row["token_address"] == "0x123"
    assert row["is_risky"] is True
    assert row["risk_score"] == 0.75


def test_write_address_csv_creates_file_with_header(tmp_path):
    records = [AddressRecord("0xaaa", 1, "sanctioned", 1.0, ["ofac"], ["ofac_sdn"], "2026-01-01T00:00:00+00:00")]
    out = tmp_path / "addr.csv"
    write_address_csv(records, out)
    assert out.exists()
    with open(out) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["address"] == "0xaaa"
    assert rows[0]["label"] == "sanctioned"


def test_write_token_csv_creates_file(tmp_path):
    records = [TokenRecord("0xbbb", 1, False, 0.0, ["goplus"], [], "2026-01-01T00:00:00+00:00")]
    out = tmp_path / "tok.csv"
    write_token_csv(records, out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_risky"] == "False"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_labels_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.labels'`

- [ ] **Step 3: Create the package file**

```python
# src/blockchain_ai/labels/__init__.py
```

- [ ] **Step 4: Create the schema module**

```python
# src/blockchain_ai/labels/schema.py
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AddressRecord:
    address: str
    chain_id: int
    label: str        # sanctioned | scammer | phishing | unknown
    confidence: float
    sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        return {
            "address": self.address,
            "chain_id": self.chain_id,
            "label": self.label,
            "confidence": self.confidence,
            "sources": "|".join(self.sources),
            "flags": "|".join(self.flags),
            "fetched_at": self.fetched_at,
        }


@dataclass
class TokenRecord:
    token_address: str
    chain_id: int
    is_risky: bool
    risk_score: float
    sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        return {
            "token_address": self.token_address,
            "chain_id": self.chain_id,
            "is_risky": self.is_risky,
            "risk_score": self.risk_score,
            "sources": "|".join(self.sources),
            "flags": "|".join(self.flags),
            "fetched_at": self.fetched_at,
        }


ADDRESS_FIELDNAMES = ["address", "chain_id", "label", "confidence", "sources", "flags", "fetched_at"]
TOKEN_FIELDNAMES = ["token_address", "chain_id", "is_risky", "risk_score", "sources", "flags", "fetched_at"]


def write_address_csv(records: list[AddressRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ADDRESS_FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_row())


def write_token_csv(records: list[TokenRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TOKEN_FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_row())
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_labels_schema.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/blockchain_ai/labels/__init__.py src/blockchain_ai/labels/schema.py tests/test_labels_schema.py
git commit -m "feat: add label schema with AddressRecord and TokenRecord"
```

---

## Task 3: GoPlus Client

**Files:**
- Create: `src/blockchain_ai/labels/goplus.py`
- Create: `tests/test_labels_goplus.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_goplus.py
from unittest.mock import patch, MagicMock
from blockchain_ai.labels.goplus import GoPlusClient
from blockchain_ai.labels.schema import AddressRecord, TokenRecord
from blockchain_ai.config import GoPlusConfig


def _client():
    return GoPlusClient("https://api.gopluslabs.io/api/v1", chain_id=1, rate_limit_per_sec=100, timeout_sec=10)


def _mock_get(json_data):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = json_data
    return m


def test_from_config():
    cfg = GoPlusConfig(base_url="https://api.gopluslabs.io/api/v1", chain_id=1, rate_limit_per_sec=2, timeout_sec=30)
    c = GoPlusClient.from_config(cfg)
    assert c._chain_id == 1


def test_get_token_security_honeypot_sets_flag():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"0xtoken1": {"is_honeypot": "1", "is_open_source": "1", "is_blacklisted": "0",
                                "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
                                "is_mintable": "0", "transfer_pausable": "0"}},
    })):
        records = _client().get_token_security(["0xtoken1"])
    assert len(records) == 1
    assert "honeypot" in records[0].flags
    assert records[0].is_risky is True
    assert records[0].sources == ["goplus"]


def test_get_token_security_safe_returns_no_flags():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"0xsafe": {"is_honeypot": "0", "is_open_source": "1", "is_blacklisted": "0",
                              "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
                              "is_mintable": "0", "transfer_pausable": "0"}},
    })):
        records = _client().get_token_security(["0xsafe"])
    assert records[0].flags == []
    assert records[0].is_risky is False


def test_get_token_security_no_source_code_is_flag():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"0xclosed": {"is_honeypot": "0", "is_open_source": "0", "is_blacklisted": "0",
                                "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
                                "is_mintable": "0", "transfer_pausable": "0"}},
    })):
        records = _client().get_token_security(["0xclosed"])
    assert "no_source_code" in records[0].flags


def test_get_address_security_phishing():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"blacklist_doubt": "0", "cybercrime": "0", "money_laundering": "0",
                   "phishing_activities": "1", "stealing_attack": "0"},
    })):
        record = _client().get_address_security("0xphisher")
    assert isinstance(record, AddressRecord)
    assert record.label == "scammer"
    assert "phishing_activities" in record.flags


def test_get_address_security_clean_returns_unknown():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"blacklist_doubt": "0", "cybercrime": "0", "money_laundering": "0",
                   "phishing_activities": "0", "stealing_attack": "0"},
    })):
        record = _client().get_address_security("0xclean")
    assert record.label == "unknown"
    assert record.flags == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_labels_goplus.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.labels.goplus'`

- [ ] **Step 3: Implement the GoPlus client**

```python
# src/blockchain_ai/labels/goplus.py
import time
import requests
from datetime import datetime, timezone
from blockchain_ai.config import GoPlusConfig
from blockchain_ai.labels.schema import AddressRecord, TokenRecord

_TOKEN_RISK_FLAGS = [
    "is_honeypot", "is_blacklisted", "can_take_back_ownership",
    "hidden_owner", "selfdestruct", "is_mintable", "transfer_pausable",
]
_ADDRESS_RISK_FLAGS = [
    "blacklist_doubt", "cybercrime", "money_laundering",
    "phishing_activities", "stealing_attack",
]
_MAX_PER_CALL = 50


class GoPlusClient:
    def __init__(self, base_url: str, chain_id: int, rate_limit_per_sec: int, timeout_sec: int):
        self._base_url = base_url.rstrip("/")
        self._chain_id = chain_id
        self._sleep_secs = 1.0 / rate_limit_per_sec
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: GoPlusConfig) -> "GoPlusClient":
        return cls(config.base_url, config.chain_id, config.rate_limit_per_sec, config.timeout_sec)

    def _get(self, path: str, params: dict | None = None) -> dict:
        time.sleep(self._sleep_secs)
        response = requests.get(f"{self._base_url}{path}", params=params or {}, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 1:
            raise RuntimeError(f"GoPlus error: {data.get('message')}")
        return data["result"]

    def get_token_security(self, addresses: list[str]) -> list[TokenRecord]:
        records: list[TokenRecord] = []
        now = datetime.now(timezone.utc).isoformat()
        for i in range(0, len(addresses), _MAX_PER_CALL):
            batch = [a.lower() for a in addresses[i: i + _MAX_PER_CALL]]
            result = self._get(f"/token_security/{self._chain_id}",
                               params={"contract_addresses": ",".join(batch)})
            for addr, data in result.items():
                flags = [f for f in _TOKEN_RISK_FLAGS if data.get(f) == "1"]
                if data.get("is_open_source") == "0":
                    flags.append("no_source_code")
                risk_score = round(len(flags) / (len(_TOKEN_RISK_FLAGS) + 1), 4)
                records.append(TokenRecord(
                    token_address=addr.lower(), chain_id=self._chain_id,
                    is_risky=risk_score > 0.1, risk_score=risk_score,
                    sources=["goplus"], flags=flags, fetched_at=now,
                ))
        return records

    def get_address_security(self, address: str) -> AddressRecord | None:
        try:
            result = self._get(f"/address_security/{address.lower()}")
        except RuntimeError:
            return None
        flags = [f for f in _ADDRESS_RISK_FLAGS if str(result.get(f, "0")) not in ("0", "")]
        confidence = round(len(flags) / len(_ADDRESS_RISK_FLAGS), 4)
        return AddressRecord(
            address=address.lower(), chain_id=self._chain_id,
            label="scammer" if flags else "unknown", confidence=confidence,
            sources=["goplus"], flags=flags,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_labels_goplus.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/labels/goplus.py tests/test_labels_goplus.py
git commit -m "feat: add GoPlus API client for token and address security"
```

---

## Task 4: OFAC Fetcher

**Files:**
- Create: `src/blockchain_ai/labels/ofac.py`
- Create: `tests/test_labels_ofac.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_ofac.py
from unittest.mock import patch, MagicMock
from blockchain_ai.labels.ofac import OFACFetcher
from blockchain_ai.config import OFACConfig

_SAMPLE_CSV = (
    "ent_num,alt_num,alt_type,alt_name,alt_remarks\n"
    "12345,1,Digital Currency Address - ETH,0xABCDEF1234567890ABCDEF1234567890ABCDEF12,\n"
    "12345,2,aka,Some Name,\n"
    "99999,1,Digital Currency Address - BTC,1BitcoinAddress,\n"
    "88888,1,Digital Currency Address - ETH,0x1111111111111111111111111111111111111111,\n"
)


def _fetcher():
    return OFACFetcher(alt_url="https://example.com/alt.csv", timeout_sec=10)


def _mock_response(text):
    m = MagicMock()
    m.text = text
    m.raise_for_status = MagicMock()
    return m


def test_from_config():
    cfg = OFACConfig(alt_url="https://example.com/alt.csv", timeout_sec=60)
    f = OFACFetcher.from_config(cfg)
    assert f._alt_url == "https://example.com/alt.csv"


def test_returns_only_eth_addresses():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    assert len(records) == 2


def test_excludes_btc_addresses():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    for r in records:
        assert r.address.startswith("0x")


def test_label_is_sanctioned_confidence_is_one():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    for r in records:
        assert r.label == "sanctioned"
        assert r.confidence == 1.0
        assert "ofac_sdn" in r.flags
        assert r.sources == ["ofac"]


def test_address_normalized_to_lowercase():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    for r in records:
        assert r.address == r.address.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_labels_ofac.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.labels.ofac'`

- [ ] **Step 3: Implement the OFAC fetcher**

```python
# src/blockchain_ai/labels/ofac.py
import csv
import io
import requests
from datetime import datetime, timezone
from blockchain_ai.config import OFACConfig
from blockchain_ai.labels.schema import AddressRecord


class OFACFetcher:
    def __init__(self, alt_url: str, timeout_sec: int):
        self._alt_url = alt_url
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: OFACConfig) -> "OFACFetcher":
        return cls(alt_url=config.alt_url, timeout_sec=config.timeout_sec)

    def fetch_eth_addresses(self) -> list[AddressRecord]:
        response = requests.get(self._alt_url, timeout=self._timeout)
        response.raise_for_status()
        reader = csv.reader(io.StringIO(response.text))
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for row in reader:
            # columns: ent_num, alt_num, alt_type, alt_name, alt_remarks
            if len(row) < 4:
                continue
            if "Digital Currency Address - ETH" not in row[2].strip():
                continue
            address = row[3].strip().lower()
            if not address.startswith("0x") or len(address) != 42:
                continue
            records.append(AddressRecord(
                address=address, chain_id=1, label="sanctioned", confidence=1.0,
                sources=["ofac"], flags=["ofac_sdn"], fetched_at=now,
            ))
        return records
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_labels_ofac.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/labels/ofac.py tests/test_labels_ofac.py
git commit -m "feat: add OFAC SDN fetcher for sanctioned ETH addresses"
```

---

## Task 5: Forta Client

**Files:**
- Create: `src/blockchain_ai/labels/forta.py`
- Create: `tests/test_labels_forta.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_forta.py
from unittest.mock import patch, MagicMock
from blockchain_ai.labels.forta import FortaClient, _severity_to_label, _severity_to_confidence
from blockchain_ai.config import FortaConfig


def _client():
    return FortaClient("https://api.forta.network/graphql", timeout_sec=10, max_alerts=100, scam_bot_ids=["0xbot1"])


def _mock_post(alerts, has_next=False):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"data": {"alerts": {
        "alerts": alerts,
        "pageInfo": {"hasNextPage": has_next, "endCursor": {"alertId": "x", "blockNumber": 1}},
    }}}
    return m


def test_from_config():
    cfg = FortaConfig(graphql_url="https://api.forta.network/graphql", timeout_sec=30, max_alerts=500, scam_bot_ids=["0xabc"])
    c = FortaClient.from_config(cfg)
    assert c._max_alerts == 500


def test_extracts_addresses_from_alerts():
    alerts = [{"hash": "0xh1", "name": "Phishing", "severity": "HIGH",
               "addresses": ["0x1111111111111111111111111111111111111111"], "createdAt": "2026-01-01"}]
    with patch("requests.post", return_value=_mock_post(alerts)):
        records = _client().fetch_scam_addresses()
    assert len(records) == 1
    assert records[0].address == "0x1111111111111111111111111111111111111111"
    assert records[0].sources == ["forta"]


def test_deduplicates_same_address_across_alerts():
    addr = "0x2222222222222222222222222222222222222222"
    alerts = [
        {"hash": "0xh1", "name": "Scam A", "severity": "HIGH", "addresses": [addr], "createdAt": "2026-01-01"},
        {"hash": "0xh2", "name": "Scam B", "severity": "CRITICAL", "addresses": [addr], "createdAt": "2026-01-01"},
    ]
    with patch("requests.post", return_value=_mock_post(alerts)):
        records = _client().fetch_scam_addresses()
    assert len(records) == 1


def test_skips_non_eth_addresses():
    alerts = [{"hash": "0xh1", "name": "Scam", "severity": "HIGH",
               "addresses": ["not_an_address", "0x3333333333333333333333333333333333333333"], "createdAt": "2026-01-01"}]
    with patch("requests.post", return_value=_mock_post(alerts)):
        records = _client().fetch_scam_addresses()
    assert len(records) == 1


def test_severity_mapping():
    assert _severity_to_label("CRITICAL") == "scammer"
    assert _severity_to_label("HIGH") == "scammer"
    assert _severity_to_label("MEDIUM") == "phishing"
    assert _severity_to_label("LOW") == "suspicious"
    assert _severity_to_confidence("CRITICAL") == 0.9
    assert _severity_to_confidence("HIGH") == 0.8
    assert _severity_to_confidence("MEDIUM") == 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_labels_forta.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.labels.forta'`

- [ ] **Step 3: Implement the Forta client**

```python
# src/blockchain_ai/labels/forta.py
import requests
from datetime import datetime, timezone
from blockchain_ai.config import FortaConfig
from blockchain_ai.labels.schema import AddressRecord

_QUERY = """
query GetAlerts($input: AlertsInput) {
  alerts(input: $input) {
    alerts { hash name severity addresses createdAt }
    pageInfo { hasNextPage endCursor { alertId blockNumber } }
  }
}
"""


class FortaClient:
    def __init__(self, graphql_url: str, timeout_sec: int, max_alerts: int, scam_bot_ids: list[str]):
        self._url = graphql_url
        self._timeout = timeout_sec
        self._max_alerts = max_alerts
        self._bot_ids = scam_bot_ids

    @classmethod
    def from_config(cls, config: FortaConfig) -> "FortaClient":
        return cls(config.graphql_url, config.timeout_sec, config.max_alerts, config.scam_bot_ids)

    def _query(self, variables: dict) -> dict:
        response = requests.post(self._url, json={"query": _QUERY, "variables": variables},
                                 headers={"Content-Type": "application/json"}, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"Forta GraphQL error: {data['errors']}")
        return data["data"]["alerts"]

    def fetch_scam_addresses(self) -> list[AddressRecord]:
        seen: dict[str, AddressRecord] = {}
        now = datetime.now(timezone.utc).isoformat()
        end_cursor = None
        collected = 0
        while collected < self._max_alerts:
            variables: dict = {"input": {"bots": self._bot_ids, "first": min(100, self._max_alerts - collected)}}
            if end_cursor:
                variables["input"]["after"] = end_cursor
            result = self._query(variables)
            alerts = result["alerts"]
            collected += len(alerts)
            for alert in alerts:
                label = _severity_to_label(alert.get("severity", ""))
                confidence = _severity_to_confidence(alert.get("severity", ""))
                flag = alert.get("name", "unknown").lower().replace(" ", "_")
                for addr in alert.get("addresses") or []:
                    addr = addr.lower()
                    if addr.startswith("0x") and len(addr) == 42 and addr not in seen:
                        seen[addr] = AddressRecord(
                            address=addr, chain_id=1, label=label, confidence=confidence,
                            sources=["forta"], flags=[flag], fetched_at=now,
                        )
            if not result["pageInfo"]["hasNextPage"]:
                break
            end_cursor = result["pageInfo"]["endCursor"]
        return list(seen.values())


def _severity_to_label(severity: str) -> str:
    return {"CRITICAL": "scammer", "HIGH": "scammer", "MEDIUM": "phishing"}.get(severity, "suspicious")


def _severity_to_confidence(severity: str) -> float:
    return {"CRITICAL": 0.9, "HIGH": 0.8, "MEDIUM": 0.6}.get(severity, 0.4)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_labels_forta.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/labels/forta.py tests/test_labels_forta.py
git commit -m "feat: add Forta GraphQL client for scam address alerts"
```

---

## Task 6: Label Unifier

**Files:**
- Create: `src/blockchain_ai/labels/unify.py`
- Create: `tests/test_labels_unify.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_unify.py
from pathlib import Path
import csv
from blockchain_ai.labels.schema import AddressRecord, write_address_csv
from blockchain_ai.labels.unify import unify_addresses


def _addr(address, label, confidence, sources, flags):
    return AddressRecord(address, 1, label, confidence, sources, flags, "2026-01-01T00:00:00+00:00")


def test_deduplicates_same_address(tmp_path):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    write_address_csv([_addr("0xaaa", "sanctioned", 1.0, ["ofac"], ["ofac_sdn"])], f1)
    write_address_csv([_addr("0xaaa", "scammer", 0.8, ["goplus"], ["phishing"]),
                       _addr("0xbbb", "unknown", 0.0, ["goplus"], [])], f2)
    out = tmp_path / "unified.csv"
    records = unify_addresses([f1, f2], out)
    assert len(records) == 2


def test_merges_flags_and_sources(tmp_path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_address_csv([_addr("0xccc", "sanctioned", 1.0, ["ofac"], ["ofac_sdn"])], f1)
    write_address_csv([_addr("0xccc", "scammer", 0.9, ["forta"], ["phishing_contract"])], f2)
    out = tmp_path / "out.csv"
    records = unify_addresses([f1, f2], out)
    r = records[0]
    assert set(r.sources) == {"ofac", "forta"}
    assert "ofac_sdn" in r.flags
    assert "phishing_contract" in r.flags


def test_keeps_highest_confidence_label(tmp_path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    write_address_csv([_addr("0xddd", "suspicious", 0.4, ["forta"], ["low"])], f1)
    write_address_csv([_addr("0xddd", "sanctioned", 1.0, ["ofac"], ["ofac_sdn"])], f2)
    out = tmp_path / "out.csv"
    records = unify_addresses([f1, f2], out)
    assert records[0].label == "sanctioned"
    assert records[0].confidence == 1.0


def test_writes_csv(tmp_path):
    f = tmp_path / "a.csv"
    write_address_csv([_addr("0xeee", "scammer", 0.8, ["goplus"], ["cybercrime"])], f)
    out = tmp_path / "out.csv"
    unify_addresses([f], out)
    assert out.exists()
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["address"] == "0xeee"


def test_skips_missing_file(tmp_path):
    out = tmp_path / "out.csv"
    records = unify_addresses([tmp_path / "missing.csv"], out)
    assert records == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_labels_unify.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.labels.unify'`

- [ ] **Step 3: Implement the unifier**

```python
# src/blockchain_ai/labels/unify.py
import csv
from pathlib import Path
from blockchain_ai.labels.schema import AddressRecord, write_address_csv


def unify_addresses(raw_paths: list[Path], output_path: Path) -> list[AddressRecord]:
    by_address: dict[str, AddressRecord] = {}
    for path in raw_paths:
        if not Path(path).exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                addr = row["address"].lower()
                sources = [s for s in row["sources"].split("|") if s]
                flags = [fl for fl in row["flags"].split("|") if fl]
                confidence = float(row["confidence"])
                if addr not in by_address:
                    by_address[addr] = AddressRecord(
                        address=addr, chain_id=int(row["chain_id"]),
                        label=row["label"], confidence=confidence,
                        sources=sources, flags=flags, fetched_at=row["fetched_at"],
                    )
                else:
                    existing = by_address[addr]
                    existing.sources = list(set(existing.sources + sources))
                    existing.flags = list(set(existing.flags + flags))
                    if confidence > existing.confidence:
                        existing.label = row["label"]
                        existing.confidence = confidence
    records = list(by_address.values())
    write_address_csv(records, output_path)
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_labels_unify.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/labels/unify.py tests/test_labels_unify.py
git commit -m "feat: add label unifier to deduplicate and merge across sources"
```

---

## Task 7: Address Classifier Config File

**Files:**
- Create: `configs/address-classifier.yaml`

- [ ] **Step 1: Create the config file**

```yaml
# configs/address-classifier.yaml
task: classification

goplus:
  base_url: https://api.gopluslabs.io/api/v1
  chain_id: 1
  rate_limit_per_sec: 2
  timeout_sec: 30

ofac:
  alt_url: https://www.treasury.gov/ofac/downloads/alt.csv
  timeout_sec: 60

forta:
  graphql_url: https://api.forta.network/graphql
  timeout_sec: 30
  max_alerts: 500
  scam_bot_ids:
    - "0x6aa2012744a3eb210fc4e4b794d9df59684d36d502fd9ebb481d58f19b917d51"
    - "0x4c7e56a9a753e29ca92bd57dd593bdab0c03e762bdd04e2bc578cb82b842c1f3"

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - tx_count
    - account_age_days
    - unique_counterparties
    - avg_tx_value_eth
    - failed_tx_ratio
    - contract_creation_count
    - erc20_token_count
    - incoming_to_outgoing_ratio
    - is_contract
    - hour_entropy
    - gas_price_avg_gwei
  fill_zero_cols:
    - failed_tx_ratio
    - contract_creation_count
    - erc20_token_count
  target_col: label

train:
  target_col: label
  model_type: xgboost
  test_size: 0.2
  hyperparameters:
    n_estimators: 300
    learning_rate: 0.05
    max_depth: 6
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42

serve:
  model_path: models/address_classifier.joblib
  confidence_threshold: 0.5
  db_path: data/jobs.db
```

- [ ] **Step 2: Verify config loads without error**

```
python -c "
import sys; sys.path.insert(0, 'src')
from blockchain_ai.config import load_config
cfg = load_config('configs/address-classifier.yaml')
print('task:', cfg.task)
print('goplus chain_id:', cfg.goplus.chain_id)
print('serve db_path:', cfg.serve.db_path)
"
```

Expected output:
```
task: classification
goplus chain_id: 1
serve db_path: data/jobs.db
```

- [ ] **Step 3: Commit**

```bash
git add configs/address-classifier.yaml
git commit -m "feat: add address-classifier.yaml unified config"
```

---

## Task 8: Etherscan Client Extensions

**Files:**
- Modify: `src/blockchain_ai/etherscan.py`
- Modify: `tests/test_etherscan.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_etherscan.py`:

```python
def test_get_tx_list_returns_list(client):
    txs = [{"hash": "0xabc", "from": "0x1", "to": "0x2", "value": "1000000000000000000",
             "isError": "0", "timeStamp": "1700000000", "gasPrice": "20000000000"}]
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(txs)
        result = client.get_tx_list("0xsome_address")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["hash"] == "0xabc"


def test_get_tx_list_returns_empty_on_no_transactions(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("No transactions found")
        result = client.get_tx_list("0xempty")
    assert result == []


def test_get_token_transfers_returns_list(client):
    transfers = [{"contractAddress": "0xtoken", "from": "0x1", "to": "0x2",
                  "value": "1000", "timeStamp": "1700000000"}]
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(transfers)
        result = client.get_token_transfers("0xsome_address")
    assert isinstance(result, list)
    assert result[0]["contractAddress"] == "0xtoken"


def test_get_token_transfers_returns_empty_on_no_transfers(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("No transactions found")
        result = client.get_token_transfers("0xempty")
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_etherscan.py::test_get_tx_list_returns_list tests/test_etherscan.py::test_get_token_transfers_returns_list -v
```

Expected: FAIL with `AttributeError: 'EtherscanClient' object has no attribute 'get_tx_list'`

- [ ] **Step 3: Add `get_tx_list` and `get_token_transfers` to `EtherscanClient`**

Append to the `EtherscanClient` class in `src/blockchain_ai/etherscan.py`:

```python
    def get_tx_list(self, address: str) -> list[dict]:
        try:
            result = self._get({
                "module": "account", "action": "txlist",
                "address": address, "startblock": 0,
                "endblock": 99999999, "sort": "asc",
            })
            return result if isinstance(result, list) else []
        except RuntimeError:
            return []

    def get_token_transfers(self, address: str) -> list[dict]:
        try:
            result = self._get({
                "module": "account", "action": "tokentx",
                "address": address, "startblock": 0,
                "endblock": 99999999, "sort": "asc",
            })
            return result if isinstance(result, list) else []
        except RuntimeError:
            return []
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_etherscan.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/etherscan.py tests/test_etherscan.py
git commit -m "feat: add get_tx_list and get_token_transfers to EtherscanClient"
```

---

## Task 9: Address Feature Extractor

**Files:**
- Create: `src/blockchain_ai/address_features.py`
- Create: `tests/test_address_features.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_address_features.py
from unittest.mock import MagicMock
from blockchain_ai.address_features import AddressFeatureExtractor, _compute_features, _entropy

_ADDRESS = "0xabc0000000000000000000000000000000000001"

_TXS = [
    {"from": _ADDRESS, "to": "0xother1", "value": "1000000000000000000",
     "isError": "0", "timeStamp": "1700000000", "gasPrice": "20000000000",
     "contractAddress": ""},
    {"from": "0xother2", "to": _ADDRESS, "value": "500000000000000000",
     "isError": "0", "timeStamp": "1700003600", "gasPrice": "25000000000",
     "contractAddress": ""},
    {"from": _ADDRESS, "to": "", "value": "0",
     "isError": "1", "timeStamp": "1700007200", "gasPrice": "15000000000",
     "contractAddress": _ADDRESS},
]

_TOKEN_TXS = [
    {"contractAddress": "0xtoken1", "from": _ADDRESS, "to": "0xother3", "value": "100", "timeStamp": "1700000000"},
    {"contractAddress": "0xtoken2", "from": "0xother4", "to": _ADDRESS, "value": "200", "timeStamp": "1700000001"},
]


def test_extract_calls_etherscan_methods():
    client = MagicMock()
    client.get_tx_list.return_value = _TXS
    client.get_token_transfers.return_value = _TOKEN_TXS
    extractor = AddressFeatureExtractor(client)
    features = extractor.extract(_ADDRESS)
    client.get_tx_list.assert_called_once_with(_ADDRESS)
    client.get_token_transfers.assert_called_once_with(_ADDRESS)
    assert isinstance(features, dict)


def test_tx_count():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["tx_count"] == 3.0


def test_failed_tx_ratio():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["failed_tx_ratio"] == pytest.approx(1/3, abs=0.001)


def test_contract_creation_count():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["contract_creation_count"] == 1.0


def test_erc20_token_count():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["erc20_token_count"] == 2.0


def test_is_contract_detected():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["is_contract"] == 1.0


def test_empty_txs_returns_zero_features():
    features = _compute_features(_ADDRESS, [], [])
    assert features["tx_count"] == 0.0
    assert features["account_age_days"] == 0.0


def test_entropy_uniform_distribution():
    values = list(range(8)) * 2  # 8 distinct hours, each appearing twice
    e = _entropy(values, 24)
    assert e == pytest.approx(3.0, abs=0.01)  # log2(8) = 3


def test_entropy_single_value():
    values = [5] * 10  # always hour 5
    e = _entropy(values, 24)
    assert e == pytest.approx(0.0, abs=0.001)


import pytest
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_address_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.address_features'`

- [ ] **Step 3: Implement the feature extractor**

```python
# src/blockchain_ai/address_features.py
import math
from datetime import datetime, timezone
from blockchain_ai.etherscan import EtherscanClient


class AddressFeatureExtractor:
    def __init__(self, client: EtherscanClient):
        self._client = client

    def extract(self, address: str) -> dict[str, float]:
        txs = self._client.get_tx_list(address)
        token_txs = self._client.get_token_transfers(address)
        return _compute_features(address.lower(), txs, token_txs)


def _compute_features(address: str, txs: list[dict], token_txs: list[dict]) -> dict[str, float]:
    now = datetime.now(timezone.utc).timestamp()
    if not txs:
        return {k: 0.0 for k in [
            "tx_count", "account_age_days", "unique_counterparties", "avg_tx_value_eth",
            "failed_tx_ratio", "contract_creation_count", "erc20_token_count",
            "incoming_to_outgoing_ratio", "is_contract", "hour_entropy", "gas_price_avg_gwei",
        ]}

    tx_count = len(txs)
    timestamps = [int(tx["timeStamp"]) for tx in txs]
    account_age_days = (now - min(timestamps)) / 86400

    counterparties: set[str] = set()
    for tx in txs:
        frm, to = tx.get("from", "").lower(), tx.get("to", "").lower()
        if frm and frm != address:
            counterparties.add(frm)
        if to and to != address:
            counterparties.add(to)

    values_eth = [int(tx.get("value", 0)) / 1e18 for tx in txs]
    avg_tx_value_eth = sum(values_eth) / tx_count

    failed = sum(1 for tx in txs if tx.get("isError") == "1")
    failed_tx_ratio = failed / tx_count

    contract_creation_count = float(sum(1 for tx in txs if not tx.get("to", "").strip()))

    incoming = sum(1 for tx in txs if tx.get("to", "").lower() == address)
    outgoing = sum(1 for tx in txs if tx.get("from", "").lower() == address)
    incoming_to_outgoing_ratio = incoming / (outgoing + 1)

    gas_prices = [int(tx["gasPrice"]) for tx in txs if tx.get("gasPrice")]
    gas_price_avg_gwei = (sum(gas_prices) / len(gas_prices) / 1e9) if gas_prices else 0.0

    hours = [datetime.fromtimestamp(int(tx["timeStamp"]), tz=timezone.utc).hour for tx in txs]
    hour_entropy = _entropy(hours, 24)

    is_contract = float(any(tx.get("contractAddress", "").lower() == address for tx in txs))

    token_contracts = set(tx.get("contractAddress", "").lower() for tx in token_txs if tx.get("contractAddress"))
    erc20_token_count = float(len(token_contracts))

    return {
        "tx_count": float(tx_count),
        "account_age_days": round(account_age_days, 2),
        "unique_counterparties": float(len(counterparties)),
        "avg_tx_value_eth": round(avg_tx_value_eth, 6),
        "failed_tx_ratio": round(failed_tx_ratio, 4),
        "contract_creation_count": contract_creation_count,
        "erc20_token_count": erc20_token_count,
        "incoming_to_outgoing_ratio": round(incoming_to_outgoing_ratio, 4),
        "is_contract": is_contract,
        "hour_entropy": round(hour_entropy, 4),
        "gas_price_avg_gwei": round(gas_price_avg_gwei, 4),
    }


def _entropy(values: list[int], n_bins: int) -> float:
    if not values:
        return 0.0
    counts = [0] * n_bins
    for v in values:
        counts[v % n_bins] += 1
    total = len(values)
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_address_features.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/address_features.py tests/test_address_features.py
git commit -m "feat: add AddressFeatureExtractor computing 11 on-chain features"
```

---

## Task 10: Job Store

**Files:**
- Create: `src/blockchain_ai/job_store.py`
- Create: `tests/test_job_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_job_store.py
import pytest
from blockchain_ai.job_store import JobStore


def test_get_returns_none_for_unknown_address(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    assert store.get("0xunknown") is None


def test_create_pending_stores_job(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xabc")
    job = store.get("0xabc")
    assert job is not None
    assert job["status"] == "pending"
    assert job["address"] == "0xabc"


def test_create_pending_is_idempotent(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xabc")
    store.create_pending("0xabc")  # second call must not raise
    assert store.get("0xabc")["status"] == "pending"


def test_mark_done_stores_result(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xdone")
    result = {"label": "scammer", "probabilities": {"sanctioned": 0.1, "scammer": 0.8, "phishing": 0.1}}
    store.mark_done("0xdone", result)
    job = store.get("0xdone")
    assert job["status"] == "done"
    import json
    stored = json.loads(job["result"])
    assert stored["label"] == "scammer"


def test_mark_failed_stores_error(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    store.create_pending("0xfail")
    store.mark_failed("0xfail", "Etherscan timeout")
    job = store.get("0xfail")
    assert job["status"] == "failed"
    assert "Etherscan timeout" in job["error"]


def test_job_store_creates_parent_dirs(tmp_path):
    store = JobStore(str(tmp_path / "nested" / "dir" / "jobs.db"))
    store.create_pending("0xtest")
    assert store.get("0xtest") is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_job_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.job_store'`

- [ ] **Step 3: Implement the job store**

```python
# src/blockchain_ai/job_store.py
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class JobStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    address    TEXT PRIMARY KEY,
                    status     TEXT NOT NULL,
                    result     TEXT,
                    error      TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def get(self, address: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT address, status, result, error, created_at, updated_at FROM jobs WHERE address = ?",
                (address,),
            ).fetchone()
        if row is None:
            return None
        return {"address": row[0], "status": row[1], "result": row[2],
                "error": row[3], "created_at": row[4], "updated_at": row[5]}

    def create_pending(self, address: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO jobs (address, status, created_at, updated_at) VALUES (?, 'pending', ?, ?)",
                (address, now, now),
            )

    def mark_done(self, address: str, result: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'done', result = ?, updated_at = ? WHERE address = ?",
                (json.dumps(result), now, address),
            )

    def mark_failed(self, address: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE address = ?",
                (error, now, address),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_job_store.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/job_store.py tests/test_job_store.py
git commit -m "feat: add SQLite-backed JobStore for async address analysis jobs"
```

---

## Task 11: Training — Classification Branch

**Files:**
- Modify: `src/blockchain_ai/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train.py`:

```python
def _classification_df():
    return pd.DataFrame({
        "tx_count": [100.0, 50.0, 200.0, 10.0, 5.0, 300.0, 80.0, 20.0, 150.0, 60.0,
                     110.0, 55.0, 210.0, 15.0, 8.0, 310.0, 85.0, 25.0, 155.0, 65.0],
        "account_age_days": [365.0, 180.0, 730.0, 30.0, 10.0, 1000.0, 400.0, 60.0, 500.0, 200.0,
                             370.0, 185.0, 735.0, 35.0, 15.0, 1005.0, 405.0, 65.0, 505.0, 205.0],
        "label": (["sanctioned"] * 7 + ["scammer"] * 7 + ["phishing"] * 6),
    })


def _classification_config():
    return TrainConfig(
        target_col="label",
        model_type="xgboost",
        stratify_col=None,
        test_size=0.2,
        hyperparameters={"n_estimators": 10, "random_state": 42},
    )


def test_train_model_classification_saves_model(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    train_model(str(csv_path), str(model_path), str(test_path), _classification_config(), task="classification")
    assert model_path.exists()


def test_train_model_classification_test_split_has_encoded_labels(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    test_path = tmp_path / "test.csv"
    train_model(str(csv_path), str(model_path), str(test_path), _classification_config(), task="classification")
    test_df = pd.read_csv(test_path)
    assert "label" in test_df.columns
    assert set(test_df["label"].unique()).issubset({0, 1, 2})


def test_train_model_classification_model_has_predict_proba(tmp_path):
    csv_path = tmp_path / "features.csv"
    _classification_df().to_csv(csv_path, index=False)
    model_path = tmp_path / "model.joblib"
    import joblib as jl
    train_model(str(csv_path), str(model_path), str(tmp_path / "t.csv"), _classification_config(), task="classification")
    model = jl.load(model_path)
    assert hasattr(model, "predict_proba")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_train.py::test_train_model_classification_saves_model -v
```

Expected: FAIL with `TypeError` (train_model doesn't accept `task` kwarg)

- [ ] **Step 3: Update `train.py` with classification branch**

Replace the full contents of `src/blockchain_ai/train.py`:

```python
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from blockchain_ai.config import TrainConfig

LABEL_TO_INT = {"sanctioned": 0, "scammer": 1, "phishing": 2}


def train_model(
    input_path: str,
    model_path: str,
    test_path: str,
    config: TrainConfig,
    task: str = "regression",
) -> object:
    df = pd.read_csv(input_path)

    if task == "classification":
        from xgboost import XGBClassifier
        X = df.drop(columns=[config.target_col])
        y = df[config.target_col].map(LABEL_TO_INT)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.hyperparameters.get("random_state", 42),
            stratify=y,
        )
        if config.model_type == "xgboost":
            hparams = {k: v for k, v in config.hyperparameters.items()}
            model = XGBClassifier(**hparams)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type!r}. Supported: 'xgboost'")
    else:
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

- [ ] **Step 4: Run all train tests**

```
pytest tests/test_train.py -v
```

Expected: all tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/train.py tests/test_train.py
git commit -m "feat: extend train_model with multi-class XGBoost classification branch"
```

---

## Task 12: Evaluation — Classification Branch

**Files:**
- Modify: `src/blockchain_ai/evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate.py`:

```python
def _make_classification_split(tmp_path):
    from xgboost import XGBClassifier
    from blockchain_ai.train import LABEL_TO_INT
    X = pd.DataFrame({"tx_count": [100.0, 50.0, 200.0, 10.0, 5.0, 300.0, 80.0, 20.0, 150.0, 60.0],
                      "age": [365.0, 180.0, 730.0, 30.0, 10.0, 1000.0, 400.0, 60.0, 500.0, 200.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 0, 1])
    model = XGBClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    model_path = tmp_path / "clf.joblib"
    joblib.dump(model, model_path)
    test_df = X.copy()
    test_df["label"] = y
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)
    return str(test_path), str(model_path)


def test_evaluate_classification_saves_report(tmp_path):
    test_path, model_path = _make_classification_split(tmp_path)
    report_path = str(tmp_path / "clf_report.json")
    import json as json_lib
    from blockchain_ai.evaluate import evaluate_model
    evaluate_model(test_path, "label", model_path, report_path, task="classification")
    assert (tmp_path / "clf_report.json").exists()
    report = json_lib.loads((tmp_path / "clf_report.json").read_text())
    assert "accuracy" in report
    assert "f1_macro" in report
    assert "f1_sanctioned" in report
    assert "f1_scammer" in report
    assert "f1_phishing" in report


def test_evaluate_classification_accuracy_is_float(tmp_path):
    test_path, model_path = _make_classification_split(tmp_path)
    import json as json_lib
    from blockchain_ai.evaluate import evaluate_model
    evaluate_model(test_path, "label", model_path, str(tmp_path / "r.json"), task="classification")
    report = json_lib.loads((tmp_path / "r.json").read_text())
    assert isinstance(report["accuracy"], float)
    assert 0.0 <= report["accuracy"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_evaluate.py::test_evaluate_classification_saves_report -v
```

Expected: FAIL with `TypeError` (evaluate_model doesn't accept `task` kwarg)

- [ ] **Step 3: Update `evaluate.py` with classification branch**

Replace the full contents of `src/blockchain_ai/evaluate.py`:

```python
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(
    test_path: str,
    target_col: str,
    model_path: str,
    report_path: str,
    task: str = "regression",
) -> dict:
    df = pd.read_csv(test_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    model = joblib.load(model_path)

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = model.predict(X)
        report = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "f1_sanctioned": float(f1_score(y, y_pred, average=None, labels=[0], zero_division=0)[0]),
            "f1_scammer": float(f1_score(y, y_pred, average=None, labels=[1], zero_division=0)[0]),
            "f1_phishing": float(f1_score(y, y_pred, average=None, labels=[2], zero_division=0)[0]),
        }
    else:
        y_log = y
        y_pred_log = model.predict(X)
        y_true = np.expm1(np.clip(np.asarray(y_log, dtype=np.float64), 0.0, 709.0))
        y_pred = np.expm1(np.clip(np.asarray(y_pred_log, dtype=np.float64), 0.0, 709.0))
        report = {
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report
```

- [ ] **Step 4: Run all evaluate tests**

```
pytest tests/test_evaluate.py -v
```

Expected: all tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/evaluate.py tests/test_evaluate.py
git commit -m "feat: extend evaluate_model with classification metrics (accuracy, F1 per class)"
```

---

## Task 13: Address Prediction Function

**Files:**
- Modify: `src/blockchain_ai/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write the failing tests**

Read `tests/test_predict.py` first, then append:

```python
import pytest
import numpy as np
from unittest.mock import MagicMock
from blockchain_ai.predict import predict_address, LABEL_ENCODER


def _mock_model(probas):
    m = MagicMock()
    m.predict_proba.return_value = np.array([probas])
    return m


def test_predict_address_returns_highest_label():
    result = predict_address(
        features={"tx_count": 10.0, "account_age_days": 30.0},
        model=_mock_model([0.05, 0.85, 0.10]),
        feature_cols=["tx_count", "account_age_days"],
        threshold=0.5,
    )
    assert result["label"] == "scammer"
    assert result["probabilities"]["scammer"] == pytest.approx(0.85, abs=0.001)


def test_predict_address_returns_unknown_below_threshold():
    result = predict_address(
        features={"tx_count": 5.0, "account_age_days": 10.0},
        model=_mock_model([0.35, 0.35, 0.30]),
        feature_cols=["tx_count", "account_age_days"],
        threshold=0.5,
    )
    assert result["label"] == "unknown"


def test_predict_address_probabilities_have_three_keys():
    result = predict_address(
        features={"tx_count": 5.0},
        model=_mock_model([0.1, 0.7, 0.2]),
        feature_cols=["tx_count"],
        threshold=0.5,
    )
    assert set(result["probabilities"].keys()) == {"sanctioned", "scammer", "phishing"}


def test_label_encoder_covers_all_classes():
    assert set(LABEL_ENCODER.values()) == {"sanctioned", "scammer", "phishing"}
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_predict.py::test_predict_address_returns_highest_label -v
```

Expected: FAIL with `ImportError` (predict_address not defined yet)

- [ ] **Step 3: Add `predict_address` and `LABEL_ENCODER` to `predict.py`**

Append to `src/blockchain_ai/predict.py`:

```python
import pandas as pd

LABEL_ENCODER: dict[int, str] = {0: "sanctioned", 1: "scammer", 2: "phishing"}


def predict_address(
    features: dict[str, float],
    model,
    feature_cols: list[str],
    threshold: float,
) -> dict:
    df = pd.DataFrame([features])[feature_cols]
    proba = model.predict_proba(df)[0]
    label_idx = int(proba.argmax())
    max_prob = float(proba.max())
    label = LABEL_ENCODER[label_idx] if max_prob >= threshold else "unknown"
    return {
        "label": label,
        "probabilities": {
            "sanctioned": round(float(proba[0]), 4),
            "scammer": round(float(proba[1]), 4),
            "phishing": round(float(proba[2]), 4),
        },
    }
```

- [ ] **Step 4: Run all predict tests**

```
pytest tests/test_predict.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/predict.py tests/test_predict.py
git commit -m "feat: add predict_address function with confidence threshold and LABEL_ENCODER"
```

---

## Task 14: Address Router

**Files:**
- Create: `src/blockchain_ai/router_address.py`
- Create: `tests/test_router_address.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router_address.py
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from blockchain_ai.job_store import JobStore
from blockchain_ai.router_address import create_router

_FEATURE_COLS = ["tx_count", "account_age_days"]


def _app(tmp_path, model=None, feature_extractor=None):
    store = JobStore(str(tmp_path / "jobs.db"))
    app = FastAPI()
    router = create_router(
        job_store=store,
        model=model,
        feature_extractor=feature_extractor,
        feature_cols=_FEATURE_COLS,
        threshold=0.5,
    )
    app.include_router(router)
    return app, store


def test_new_address_returns_202(tmp_path):
    app, _ = _app(tmp_path)
    resp = TestClient(app).get("/predict/address/0x1234567890abcdef1234567890abcdef12345678")
    assert resp.status_code == 202
    assert resp.json()["status"] == "pending"


def test_pending_address_returns_202(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xabc")
    resp = TestClient(app).get("/predict/address/0xabc")
    assert resp.status_code == 202


def test_done_address_returns_200_with_result(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xdone")
    result = {"label": "scammer", "probabilities": {"sanctioned": 0.1, "scammer": 0.8, "phishing": 0.1}}
    store.mark_done("0xdone", result)
    resp = TestClient(app).get("/predict/address/0xdone")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "done"
    assert data["label"] == "scammer"
    assert "probabilities" in data


def test_failed_address_returns_200_with_error(tmp_path):
    app, store = _app(tmp_path)
    store.create_pending("0xfail")
    store.mark_failed("0xfail", "Etherscan timeout")
    resp = TestClient(app).get("/predict/address/0xfail")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "failed"
    assert "Etherscan timeout" in data["error"]


def test_address_normalized_to_lowercase(tmp_path):
    app, store = _app(tmp_path)
    TestClient(app).get("/predict/address/0xABCDEF1234567890ABCDEF1234567890ABCDEF12")
    job = store.get("0xabcdef1234567890abcdef1234567890abcdef12")
    assert job is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_router_address.py -v
```

Expected: `ModuleNotFoundError: No module named 'blockchain_ai.router_address'`

- [ ] **Step 3: Implement the router**

```python
# src/blockchain_ai/router_address.py
import json
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from blockchain_ai.job_store import JobStore
from blockchain_ai.predict import predict_address


def create_router(
    job_store: JobStore,
    model,
    feature_extractor,
    feature_cols: list[str],
    threshold: float,
) -> APIRouter:
    router = APIRouter()

    @router.get("/predict/address/{address}")
    def predict_address_endpoint(address: str, background_tasks: BackgroundTasks):
        address = address.lower()
        job = job_store.get(address)
        if job is None:
            job_store.create_pending(address)
            background_tasks.add_task(_run_job, address, job_store, model, feature_extractor, feature_cols, threshold)
            return JSONResponse({"address": address, "status": "pending"}, status_code=202)
        if job["status"] == "pending":
            return JSONResponse({"address": address, "status": "pending"}, status_code=202)
        if job["status"] == "done":
            result = json.loads(job["result"])
            return {"address": address, "status": "done", **result}
        return {"address": address, "status": "failed", "error": job.get("error")}

    return router


def _run_job(
    address: str,
    job_store: JobStore,
    model,
    feature_extractor,
    feature_cols: list[str],
    threshold: float,
) -> None:
    try:
        if feature_extractor is None:
            raise RuntimeError("Etherscan client not available")
        if model is None:
            raise RuntimeError("Model not loaded")
        features = feature_extractor.extract(address)
        result = predict_address(features, model, feature_cols, threshold)
        job_store.mark_done(address, result)
    except Exception as exc:
        job_store.mark_failed(address, str(exc))
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_router_address.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/router_address.py tests/test_router_address.py
git commit -m "feat: add address classification router with async job pattern"
```

---

## Task 15: App.py — Classification Branch

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `app.py` with task-branching version**

```python
#!/usr/bin/env python3
"""
FastAPI server built from a pipeline YAML config.
Usage:
  CONFIG=configs/ethereum-gas-price.yaml uvicorn app:app --reload
  CONFIG=configs/address-classifier.yaml uvicorn app:app --reload
"""
import io
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import Field, create_model

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))
from blockchain_ai.config import FieldConfig, ServeConfig, load_config
from blockchain_ai.etherscan import EtherscanClient

_CONFIG_PATH = os.environ.get("CONFIG", "configs/ethereum-gas-price.yaml")
_cfg = load_config(_CONFIG_PATH)

if _cfg.serve is None:
    raise RuntimeError(f"Config at {_CONFIG_PATH} is missing a 'serve' section.")

task = _cfg.task
serve: ServeConfig = _cfg.serve
feature_cols: list[str] = _cfg.ingest.feature_cols

if task == "regression":
    app = FastAPI(title=serve.title, description=serve.description, version="0.1.0")
else:
    app = FastAPI(title="Ethereum Address Classifier", description="Classifies Ethereum addresses.", version="0.1.0")

_etherscan_client = None
if _cfg.etherscan is not None:
    try:
        _etherscan_client = EtherscanClient.from_config(_cfg.etherscan)
    except Exception as exc:
        print(f"WARNING: Etherscan client could not be initialized ({exc}).")

_raw_model_path = os.environ.get("MODEL_PATH", serve.model_path)
model = None
try:
    if _raw_model_path.startswith("gs://"):
        import tempfile
        from google.cloud import storage as gcs
        _tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
        _bucket_name, _, _blob_path = _raw_model_path[5:].partition("/")
        gcs.Client().bucket(_bucket_name).blob(_blob_path).download_to_filename(_tmp.name)
        _model_path = _tmp.name
    else:
        _model_path = _raw_model_path
    model = joblib.load(_model_path)
except Exception as exc:
    print(f"WARNING: model could not be loaded ({exc}). Prediction endpoints will return 503.")


@app.get("/health")
def health():
    return {"status": "ok"}


if task == "regression":
    _TREND_LOOKBACK = 10

    def _fetch_latest_features() -> pd.DataFrame:
        if _etherscan_client is None:
            raise HTTPException(status_code=503, detail="Etherscan client not available. Check ETHERSCAN_API_KEY and etherscan config.")
        latest = _etherscan_client.get_latest_block_number()
        rows = []
        for block_num in range(latest - _TREND_LOOKBACK, latest + 1):
            row = _etherscan_client.get_block(block_num)
            if row:
                rows.append(row)
        if not rows:
            raise HTTPException(status_code=503, detail="Could not fetch recent blocks from Etherscan.")
        df = pd.DataFrame(rows)
        df["base_fee_gwei"] = df["base_fee_per_gas"] / 1e9
        df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.dayofweek
        shifted = df["base_fee_gwei"].shift(_TREND_LOOKBACK)
        df["base_fee_trend"] = ((df["base_fee_gwei"] - shifted) / shifted).fillna(0.0)
        return df

    _TYPE_MAP = {"float": float, "int": int}

    def _pydantic_field(fc: FieldConfig) -> Any:
        constraints: dict[str, Any] = {"description": fc.description, "examples": [fc.example]}
        if fc.ge is not None: constraints["ge"] = fc.ge
        if fc.gt is not None: constraints["gt"] = fc.gt
        if fc.le is not None: constraints["le"] = fc.le
        if fc.lt is not None: constraints["lt"] = fc.lt
        return (_TYPE_MAP[fc.type], Field(**constraints))

    TransactionModel = create_model(
        "Transaction",
        **{name: _pydantic_field(fc) for name, fc in serve.fields.items()},
    )

    def _to_response(gwei: float) -> dict:
        return {
            f"predicted_{serve.target_description.lower().replace(' ', '_')}_wei": gwei * 1e9,
            f"predicted_{serve.target_description.lower().replace(' ', '_')}_gwei": gwei,
        }

    def _predict_df(df: pd.DataFrame) -> np.ndarray:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not available yet. The retrain job may not have run.")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")
        raw = model.predict(df[feature_cols])
        return np.expm1(raw) if serve.log_transform else raw

    def _predict_n_blocks(df: pd.DataFrame, n: int) -> list[dict]:
        last = df.iloc[-1]
        last_block = int(last["block_number"])
        last_timestamp = float(last["timestamp"])
        gas_used_ratio = float(last["gas_used_ratio"])
        rolling_fees: list[float] = list(df["base_fee_gwei"].values)
        predictions = []
        for step in range(1, n + 1):
            prev_fee = rolling_fees[-1]
            block_number = last_block + step
            timestamp = last_timestamp + step * 12
            if step == 1:
                base_fee = prev_fee * (1 + (gas_used_ratio - 0.5) / 4)
                method = "formula"
            else:
                dt = pd.Timestamp(timestamp, unit="s", tz="UTC")
                lookback = len(rolling_fees) - 1 - _TREND_LOOKBACK
                trend = (prev_fee - rolling_fees[lookback]) / rolling_fees[lookback] if lookback >= 0 and rolling_fees[lookback] > 0 else 0.0
                features = pd.DataFrame([{"base_fee_gwei": prev_fee, "gas_used_ratio": gas_used_ratio,
                                          "hour_of_day": dt.hour, "day_of_week": dt.dayofweek, "base_fee_trend": trend}])
                base_fee = float(_predict_df(features)[0])
                method = "model"
            rolling_fees.append(base_fee)
            predictions.append({"step": step, "block_number": block_number, "base_fee_gwei": base_fee,
                                 "base_fee_wei": base_fee * 1e9, "method": method})
        return predictions

    @app.post("/predict", summary=f"Predict {serve.target_description} for a single transaction")
    def predict_json(tx: TransactionModel):  # type: ignore[valid-type]
        df = pd.DataFrame([tx.model_dump()])[feature_cols]
        preds = _predict_df(df)
        return _to_response(float(preds[0]))

    @app.post("/predict/batch", summary=f"Predict {serve.target_description} for multiple transactions via CSV",
              description=f"Upload a CSV with columns: `{'`, `'.join(feature_cols)}`. Returns predictions in the same row order.")
    async def predict_csv(file: UploadFile = File(..., description="CSV file with transaction rows.")):
        if not (file.filename or "").endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {e}")
        preds = _predict_df(df)
        return JSONResponse({"count": len(preds), "predictions": [_to_response(float(w)) for w in preds.tolist()]})

    @app.get("/predict/latest", summary=f"Predict {serve.target_description} using live on-chain data",
             description="Fetches the latest block from Etherscan, computes all features automatically, and returns a prediction. No input required.")
    def predict_latest(n_blocks: int = Query(default=1, ge=1, le=50)):
        df = _fetch_latest_features()
        return {
            "block_number": int(df["block_number"].iloc[-1]),
            "block_history": df[["block_number", "base_fee_gwei"]].rename(columns={"block_number": "block"}).to_dict(orient="records"),
            "predictions": _predict_n_blocks(df, n_blocks),
        }

elif task == "classification":
    from blockchain_ai.job_store import JobStore
    from blockchain_ai.router_address import create_router
    from blockchain_ai.address_features import AddressFeatureExtractor

    _job_store = JobStore(serve.db_path)
    _feature_extractor = AddressFeatureExtractor(_etherscan_client) if _etherscan_client else None
    _router = create_router(
        job_store=_job_store,
        model=model,
        feature_extractor=_feature_extractor,
        feature_cols=feature_cols,
        threshold=serve.confidence_threshold,
    )
    app.include_router(_router)
```

- [ ] **Step 2: Verify regression app still starts**

```bash
CONFIG=configs/ethereum-gas-price.yaml python -c "import app; print('regression app OK')"
```

Expected: `regression app OK` (may warn about missing model/API key, that's fine)

- [ ] **Step 3: Verify classification app starts**

```bash
CONFIG=configs/address-classifier.yaml python -c "import app; print('classification app OK')"
```

Expected: `classification app OK`

- [ ] **Step 4: Run full test suite to ensure nothing is broken**

```
pytest tests/ -v --ignore=tests/test_streamlit_ui.py
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: extend app.py to serve address classifier when task=classification"
```

---

## Task 16: Training Scripts

**Files:**
- Create: `scripts/collect_labels.py`
- Create: `scripts/collect_address_features.py`
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Create `scripts/collect_labels.py`**

```python
#!/usr/bin/env python3
"""
Collect ground-truth address labels from GoPlus, OFAC, and Forta, then unify them.

Usage:
    python scripts/collect_labels.py --config configs/address-classifier.yaml
    python scripts/collect_labels.py --config configs/address-classifier.yaml \
        --addresses 0xABC...,0xDEF...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.labels.goplus import GoPlusClient
from blockchain_ai.labels.ofac import OFACFetcher
from blockchain_ai.labels.forta import FortaClient
from blockchain_ai.labels.schema import write_address_csv
from blockchain_ai.labels.unify import unify_addresses


def main():
    parser = argparse.ArgumentParser(description="Collect address labels from GoPlus, OFAC, and Forta")
    parser.add_argument("--config", default="configs/address-classifier.yaml")
    parser.add_argument("--addresses", default="", help="Comma-separated addresses for GoPlus address security")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path("data/raw/labels")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed/labels")
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = []

    if args.addresses and cfg.goplus:
        print("[goplus] Fetching address security...")
        client = GoPlusClient.from_config(cfg.goplus)
        addr_list = [a.strip() for a in args.addresses.split(",") if a.strip()]
        records = [r for a in addr_list if (r := client.get_address_security(a)) is not None]
        path = raw_dir / "goplus_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    if cfg.ofac:
        print("[ofac] Fetching sanctioned ETH addresses...")
        records = OFACFetcher.from_config(cfg.ofac).fetch_eth_addresses()
        path = raw_dir / "ofac_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    if cfg.forta:
        print("[forta] Fetching scam alerts...")
        records = FortaClient.from_config(cfg.forta).fetch_scam_addresses()
        path = raw_dir / "forta_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    print("[unify] Merging and deduplicating...")
    unified = unify_addresses(raw_files, processed_dir / "addresses.csv")
    print(f"  Unified {len(unified)} unique addresses → {processed_dir / 'addresses.csv'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/collect_address_features.py`**

```python
#!/usr/bin/env python3
"""
Fetch Etherscan on-chain features for all labeled addresses.
Run after collect_labels.py.

Usage:
    python scripts/collect_address_features.py --config configs/address-classifier.yaml
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.etherscan import EtherscanClient
from blockchain_ai.address_features import AddressFeatureExtractor

LABELS_PATH = "data/processed/labels/addresses.csv"
OUTPUT_PATH = "data/processed/features/address_features.csv"


def main():
    parser = argparse.ArgumentParser(description="Collect Etherscan features for labeled addresses")
    parser.add_argument("--config", default="configs/address-classifier.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")

    labels_path = Path(LABELS_PATH)
    if not labels_path.exists():
        raise FileNotFoundError(f"{LABELS_PATH} not found. Run collect_labels.py first.")

    with open(labels_path) as f:
        labeled = [(row["address"], row["label"]) for row in csv.DictReader(f)]

    print(f"Found {len(labeled)} labeled addresses. Fetching features...")

    extractor = AddressFeatureExtractor(EtherscanClient.from_config(cfg.etherscan))
    feature_cols = cfg.ingest.feature_cols
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (address, label) in enumerate(labeled):
        print(f"  [{i + 1}/{len(labeled)}] {address}")
        try:
            features = extractor.extract(address)
            row = {col: features.get(col, 0.0) for col in feature_cols}
            row["label"] = label
            rows.append(row)
        except Exception as e:
            print(f"    WARNING: Failed for {address}: {e}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_cols + ["label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Update `scripts/run_pipeline.py` with classification branch**

Replace the full contents of `scripts/run_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Run the full ML pipeline: ingest/features -> [hpo] -> train -> evaluate.

Regression usage:
    poetry run python scripts/run_pipeline.py \
        --input data/raw/ethereum-blocks.csv \
        --config configs/ethereum-gas-price.yaml

Classification usage (features CSV must already exist from collect_address_features.py):
    poetry run python scripts/run_pipeline.py \
        --config configs/address-classifier.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--input", default=None, help="Path to raw input CSV (required for regression)")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg.task

    if task == "regression":
        if not args.input:
            parser.error("--input is required for task=regression")
        from blockchain_ai.ingest import load_and_clean
        from blockchain_ai.tune import run_hpo

        processed_path = "data/processed/ethereum-transactions.csv"
        test_path = "data/processed/ethereum-transactions-test.csv"
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
            print("[2/4] Skipping HPO (no hpo section in config)")

        print(f"[3/4] Training {train_config.model_type} model ...")
        train_model(processed_path, model_path, test_path, train_config)

        print("[4/4] Evaluating model ...")
        report = evaluate_model(test_path, train_config.target_col, model_path, report_path)

    else:
        features_path = "data/processed/features/address_features.csv"
        model_path = cfg.serve.model_path if cfg.serve else "models/address_classifier.joblib"
        test_path = "data/processed/address_features_test.csv"
        report_path = "reports/address_classifier_report.json"

        if not Path(features_path).exists():
            raise FileNotFoundError(f"{features_path} not found. Run collect_address_features.py first.")

        print(f"[1/2] Training classification model from {features_path} ...")
        train_model(features_path, model_path, test_path, cfg.train, task="classification")

        print("[2/2] Evaluating model ...")
        report = evaluate_model(test_path, cfg.train.target_col, model_path, report_path, task="classification")

    print(f"\nPipeline complete. Report:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify scripts have no import errors**

```bash
python -c "import sys; sys.path.insert(0,'src'); exec(open('scripts/collect_labels.py').read().replace('main()',''))"
python -c "import sys; sys.path.insert(0,'src'); exec(open('scripts/collect_address_features.py').read().replace('main()',''))"
python -c "import sys; sys.path.insert(0,'src'); exec(open('scripts/run_pipeline.py').read().replace('main()',''))"
```

Expected: no errors

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v --ignore=tests/test_streamlit_ui.py
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_labels.py scripts/collect_address_features.py scripts/run_pipeline.py
git commit -m "feat: add label/feature collection scripts and classification pipeline"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|-------------|------|
| Unified YAML config with task + goplus/ofac/forta | Task 1, 7 |
| GoPlus token + address security | Task 3 |
| OFAC sanctioned addresses | Task 4 |
| Forta scam alerts | Task 5 |
| Label deduplication/unification | Task 6 |
| Etherscan txlist + tokentx | Task 8 |
| 11 on-chain features | Task 9 |
| SQLite job store | Task 10 |
| XGBoost multi-class training | Task 11 |
| Per-class F1 evaluation | Task 12 |
| predict_address with threshold | Task 13 |
| Async GET /predict/address/{address} | Task 14 |
| app.py branches on task | Task 15 |
| Offline training scripts | Task 16 |
| 202/200 response shapes | Task 14 |
| `unknown` returned below threshold | Task 13, 14 |

### Placeholder scan

No TBDs, no "implement later", no stubs. All code blocks are complete.

### Type consistency

- `GoPlusConfig`, `OFACConfig`, `FortaConfig` defined in Task 1, used via `from_config()` in Tasks 3–5 ✓
- `LABEL_TO_INT` defined in Task 11 (`train.py`), `LABEL_ENCODER` defined in Task 13 (`predict.py`) — consistent inverse mapping ✓
- `create_router(job_store, model, feature_extractor, feature_cols, threshold)` signature in Task 14 matches call in Task 15 ✓
- `AddressFeatureExtractor.extract()` returns `dict[str, float]`, consumed by `predict_address(features, model, feature_cols, threshold)` in Task 13 ✓
- `train_model(..., task="classification")` and `evaluate_model(..., task="classification")` signatures added in Tasks 11–12, called correctly in Task 16 ✓
