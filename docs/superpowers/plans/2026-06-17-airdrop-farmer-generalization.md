# Airdrop Farmer Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the airdrop farmer detector so training can pull from multiple known airdrop contracts, and serving scores any wallet address with no contract/date needed at request time — anchoring features on the wallet's own history and tracking shared-funder reuse via a persistent, cross-airdrop ledger.

**Architecture:** `AirdropConfig` moves from a single `contract_address`/`date` to a list of `contract_addresses` used only by the seed step to build a training pool. `compute_airdrop_features` drops its `contract_address`/`airdrop_date` params and instead anchors two renamed features on the wallet's own first token receipt; a new `FunderLedger` (sqlite, same style as `JobStore`) replaces the per-run `funding_address_set`, persisting `(funder, wallet)` pairs across both the seed step and live serving so the shared-funder signal compounds over time. `GMMWrapper` drops the `funding_address_set` field entirely since that state now lives outside the model artifact.

**Tech Stack:** Python 3.12, scikit-learn (GaussianMixture, StandardScaler), SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-06-17-airdrop-farmer-generalization-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/blockchain_ai/config.py` | `AirdropConfig.contract_addresses: list[str]`, drop `date` |
| Create | `src/blockchain_ai/database/funder_ledger.py` | Persistent cross-airdrop funder→wallet ledger |
| Modify | `src/blockchain_ai/feature/airdrop_features.py` | Event-agnostic feature extraction + ledger integration |
| Modify | `src/blockchain_ai/model/gmm_wrapper.py` | Drop `funding_address_set`, rename farmer-signal feature keys |
| Modify | `configs/airdrop-farmer.yaml` | `contract_addresses` list, renamed `feature_cols` |
| Modify | `app.py` | Wire `FunderLedger` into `AirdropFeatureExtractor`, drop date/contract args |
| Modify | `src/blockchain_ai/workflow/run_airdrop_farmer_seed.py` | Loop over multiple contracts, two-pass ledger recording |
| Modify | `tests/test_config.py` | Update airdrop config tests for `contract_addresses` |
| Create | `tests/test_funder_ledger.py` | Unit tests for `FunderLedger` |
| Modify | `tests/test_airdrop_features.py` | Rewrite for new signature and anchor logic |
| Modify | `tests/test_gmm_wrapper.py` | Rename feature keys, drop `funding_address_set` test |
| Modify | `tests/test_router_airdrop_farmer.py` | Rename feature keys in fixtures |

---

## Task 1: Config — `AirdropConfig.contract_addresses`

**Files:**
- Modify: `src/blockchain_ai/config.py:104-107` (dataclass), `src/blockchain_ai/config.py:314-323` (parsing)
- Modify: `tests/test_config.py:428-463`

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py`, replace the `test_airdrop_farmer_config_parsed` function (lines 428-463) with:

```python
def test_airdrop_farmer_config_parsed(tmp_path):
    yaml_content = """
task: airdrop_farmer

airdrop:
  contract_addresses:
    - "0xABC"
    - "0xDEF"

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - wallet_age_days
  fill_zero_cols: []

train:
  model_type: gmm
  hyperparameters:
    n_components: 4

serve:
  model_path: models/gmm.joblib
  db_path: data/jobs.db
"""
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_content)
    cfg = load_config(str(p))
    assert cfg.task == "airdrop_farmer"
    assert cfg.airdrop is not None
    assert cfg.airdrop.contract_addresses == ["0xABC", "0xDEF"]
    assert cfg.serve.db_path == "data/jobs.db"
```

Leave `test_airdrop_farmer_missing_airdrop_section_raises` (lines 466-493) untouched — it doesn't reference `contract_address`/`date`.

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_airdrop_farmer_config_parsed -v
```

Expected: FAIL with `AttributeError: 'AirdropConfig' object has no attribute 'contract_addresses'`.

- [ ] **Step 3: Update the `AirdropConfig` dataclass**

In `src/blockchain_ai/config.py:104-107`, replace:

```python
@dataclass
class AirdropConfig:
    contract_address: str
    date: str  # ISO format e.g. "2024-01-15"
```

with:

```python
@dataclass
class AirdropConfig:
    contract_addresses: list[str]
```

- [ ] **Step 4: Update the parsing block in `load_config`**

In `src/blockchain_ai/config.py:314-323`, replace:

```python
    airdrop_cfg = None
    if "airdrop" in raw:
        a = raw["airdrop"]
        for key in ("contract_address", "date"):
            if key not in a:
                raise ValueError(f"Config airdrop section missing required key: '{key}'")
        airdrop_cfg = AirdropConfig(
            contract_address=a["contract_address"],
            date=a["date"],
        )
```

with:

```python
    airdrop_cfg = None
    if "airdrop" in raw:
        a = raw["airdrop"]
        if "contract_addresses" not in a:
            raise ValueError("Config airdrop section missing required key: 'contract_addresses'")
        airdrop_cfg = AirdropConfig(
            contract_addresses=list(a["contract_addresses"]),
        )
```

(The lines immediately after — `if task == "airdrop_farmer" and airdrop_cfg is None: raise ValueError(...)` — are unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
```

Expected: all tests PASS, including both airdrop tests.

- [ ] **Step 6: Commit**

```bash
git add src/blockchain_ai/config.py tests/test_config.py
git commit -m "feat: support multiple airdrop contract addresses in config"
```

---

## Task 2: `FunderLedger`

**Files:**
- Create: `src/blockchain_ai/database/funder_ledger.py`
- Create: `tests/test_funder_ledger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_funder_ledger.py`:

```python
from blockchain_ai.database.funder_ledger import FunderLedger


def test_funded_count_zero_for_unknown_funder(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    assert ledger.funded_count("0xnobody", "0xwalleta") == 0


def test_record_then_funded_count_excludes_self(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xfunder", "0xwalleta")
    assert ledger.funded_count("0xfunder", "0xwalleta") == 0


def test_funded_count_counts_other_distinct_wallets(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xfunder", "0xwalleta")
    ledger.record("0xfunder", "0xwalletb")
    ledger.record("0xfunder", "0xwalletc")
    assert ledger.funded_count("0xfunder", "0xwalleta") == 2


def test_record_is_idempotent(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xfunder", "0xwalleta")
    ledger.record("0xfunder", "0xwalleta")
    ledger.record("0xfunder", "0xwalletb")
    assert ledger.funded_count("0xfunder", "0xwalletb") == 1


def test_addresses_are_normalized_to_lowercase(tmp_path):
    ledger = FunderLedger(str(tmp_path / "ledger.db"))
    ledger.record("0xFunder", "0xWalletA")
    assert ledger.funded_count("0XFUNDER", "0xother") == 1


def test_ledger_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "ledger.db")
    FunderLedger(db_path).record("0xfunder", "0xwalleta")
    ledger2 = FunderLedger(db_path)
    assert ledger2.funded_count("0xfunder", "0xwalletb") == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_funder_ledger.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'blockchain_ai.database.funder_ledger'`.

- [ ] **Step 3: Implement `FunderLedger`**

Create `src/blockchain_ai/database/funder_ledger.py`:

```python
import sqlite3
from pathlib import Path


class FunderLedger:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS funder_ledger (
                    funder TEXT NOT NULL,
                    wallet TEXT NOT NULL,
                    PRIMARY KEY (funder, wallet)
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def record(self, funder: str, wallet: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO funder_ledger (funder, wallet) VALUES (?, ?)",
                (funder.lower(), wallet.lower()),
            )

    def funded_count(self, funder: str, exclude_wallet: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT wallet) FROM funder_ledger WHERE funder = ? AND wallet != ?",
                (funder.lower(), exclude_wallet.lower()),
            ).fetchone()
        return int(row[0]) if row else 0
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_funder_ledger.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/database/funder_ledger.py tests/test_funder_ledger.py
git commit -m "feat: add persistent cross-airdrop FunderLedger"
```

---

## Task 3: Event-Agnostic `airdrop_features.py`

**Files:**
- Modify: `src/blockchain_ai/feature/airdrop_features.py` (full rewrite)
- Modify: `tests/test_airdrop_features.py` (full rewrite)

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_airdrop_features.py` with:

```python
import math
from unittest.mock import MagicMock
from blockchain_ai.feature.airdrop_features import compute_airdrop_features, derive_funder, AirdropFeatureExtractor

_ADDR = "0xwallet000000000000000000000000000000001"
_BASE_TS = 1_700_000_000  # 2023-11-14

# wallet that received a token and immediately dumped it — sybil pattern
_TXS_SYBIL = [
    # funded by a shared funder
    {"from": "0xfunder", "to": _ADDR, "value": "1000000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 100), "gasPrice": "20000000000"},
    # unrelated activity after receiving the token
    {"from": _ADDR, "to": "0xexchange", "value": "0",
     "isError": "0", "timeStamp": str(_BASE_TS + 120), "gasPrice": "20000000000"},
]

_TOKEN_TXS_SYBIL = [
    # received token1 at _BASE_TS + 60
    {"contractAddress": "0xtoken1", "from": "0xdistributor", "to": _ADDR,
     "value": "1000", "timeStamp": str(_BASE_TS + 60)},
    # dumped token1 30 minutes later
    {"contractAddress": "0xtoken1", "from": _ADDR, "to": "0xexchange",
     "value": "1000", "timeStamp": str(_BASE_TS + 1860)},
]

# wallet with long history that held its token for months — genuine pattern
_TXS_GENUINE = [
    {"from": "0xother1", "to": _ADDR, "value": "500000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 86400 * 365), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother2", "value": "100000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 86400 * 300), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother3", "value": "100000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 86400 * 200), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother4", "value": "0",
     "isError": "0", "timeStamp": str(_BASE_TS + 86400 * 10), "gasPrice": "20000000000"},
]

_TOKEN_TXS_GENUINE = [
    # first-ever token receipt: token1 at -100 days (anchor for tx_count_before_first_inflow)
    {"contractAddress": "0xtoken1", "from": "0xother1", "to": _ADDR,
     "value": "1000", "timeStamp": str(_BASE_TS - 86400 * 100)},
    # second token received later
    {"contractAddress": "0xtoken2", "from": "0xother2", "to": _ADDR,
     "value": "2000", "timeStamp": str(_BASE_TS - 86400 * 50)},
    # token1 finally sent out 140 days after it was received
    {"contractAddress": "0xtoken1", "from": _ADDR, "to": "0xexchange",
     "value": "500", "timeStamp": str(_BASE_TS + 86400 * 40)},
]

_NO_FUNDER_LOOKUP = lambda funder: 0  # noqa: E731


def test_wallet_age_days_sybil_is_short():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    assert f["wallet_age_days"] < 1


def test_wallet_age_days_genuine_is_long():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    assert f["wallet_age_days"] > 300


def test_tx_count_before_first_inflow_sybil_is_one():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    # only the funding tx at _BASE_TS - 100 is before the token receipt at _BASE_TS + 60
    assert f["tx_count_before_first_inflow"] == 1


def test_tx_count_before_first_inflow_genuine_is_three():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    # three txs happen before the first token receipt at -100 days
    assert f["tx_count_before_first_inflow"] == 3


def test_tx_count_before_first_inflow_no_token_receipt_uses_total_tx_count():
    txs = [
        {"from": _ADDR, "to": "0xa", "value": "0", "isError": "0",
         "timeStamp": str(_BASE_TS), "gasPrice": "1"},
        {"from": _ADDR, "to": "0xb", "value": "0", "isError": "0",
         "timeStamp": str(_BASE_TS + 60), "gasPrice": "1"},
    ]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["tx_count_before_first_inflow"] == 2.0


def test_token_type_diversity():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    assert f["token_type_diversity"] == 1

    f2 = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    assert f2["token_type_diversity"] == 2


def test_inflow_to_outflow_hours_sybil_is_short():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    # received at +60, dumped at +1860 → 1800s = 0.5 hours
    assert abs(f["inflow_to_outflow_hours"] - 0.5) < 0.01


def test_inflow_to_outflow_hours_genuine_is_long():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    # token1 received at -100 days, sent at +40 days → 140 days = 3360 hours
    assert f["inflow_to_outflow_hours"] > 3000


def test_inflow_to_outflow_hours_no_token_activity_returns_zero():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
            "isError": "0", "timeStamp": str(_BASE_TS + 100), "gasPrice": "20000000000"}]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["inflow_to_outflow_hours"] == 0.0


def test_shared_funder_score_uses_log1p_of_lookup():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, lambda funder: 3)
    assert abs(f["shared_funder_score"] - math.log1p(3)) < 1e-9


def test_shared_funder_score_zero_when_lookup_returns_zero():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    assert f["shared_funder_score"] == 0.0


def test_shared_funder_score_zero_when_no_funder():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
            "isError": "0", "timeStamp": str(_BASE_TS + 100), "gasPrice": "20000000000"}]
    f = compute_airdrop_features(_ADDR, txs, [], lambda funder: 5)
    assert f["shared_funder_score"] == 0.0


def test_inter_tx_time_variance_single_tx_is_zero():
    txs = [_TXS_SYBIL[0]]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["inter_tx_time_variance"] == 0.0


def test_inter_tx_time_variance_regular_spacing_is_zero():
    base = _BASE_TS
    txs = [
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base), "gasPrice": "20000000000"},
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base + 60), "gasPrice": "20000000000"},
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base + 120), "gasPrice": "20000000000"},
    ]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["inter_tx_time_variance"] == 0.0


def test_unique_counterparty_count():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    # counterparties: 0xfunder, 0xexchange = 2
    assert f["unique_counterparty_count"] == 2.0


def test_derive_funder_returns_earliest_inbound_value_sender():
    assert derive_funder(_ADDR, _TXS_SYBIL) == "0xfunder"


def test_derive_funder_returns_none_when_no_inbound_value():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
            "isError": "0", "timeStamp": str(_BASE_TS), "gasPrice": "1"}]
    assert derive_funder(_ADDR, txs) is None


def test_extractor_calls_etherscan_records_funder_and_delegates():
    client = MagicMock()
    client.get_tx_list.return_value = _TXS_SYBIL
    client.get_token_transfers.return_value = _TOKEN_TXS_SYBIL
    ledger = MagicMock()
    ledger.funded_count.return_value = 0
    extractor = AirdropFeatureExtractor(client, ledger)
    features = extractor.extract(_ADDR)
    client.get_tx_list.assert_called_once_with(_ADDR)
    client.get_token_transfers.assert_called_once_with(_ADDR)
    ledger.record.assert_called_once_with("0xfunder", _ADDR)
    assert "wallet_age_days" in features


def test_empty_txs_returns_zero_features():
    f = compute_airdrop_features(_ADDR, [], [], _NO_FUNDER_LOOKUP)
    assert all(v == 0.0 for v in f.values())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_airdrop_features.py -v
```

Expected: FAIL — `ImportError: cannot import name 'derive_funder' from 'blockchain_ai.feature.airdrop_features'` (current file doesn't have this function or the new signature).

- [ ] **Step 3: Rewrite `airdrop_features.py`**

Replace the entire contents of `src/blockchain_ai/feature/airdrop_features.py` with:

```python
import math
from datetime import datetime, timezone
from typing import Callable

from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.database.funder_ledger import FunderLedger
from blockchain_ai.feature.feature_extractor import FeatureExtractor


class AirdropFeatureExtractor(FeatureExtractor):
    def __init__(self, client: EtherscanClient, funder_ledger: FunderLedger):
        self._client = client
        self._funder_ledger = funder_ledger

    def extract(self, address: str) -> dict[str, float]:
        address = address.lower()
        txs = self._client.get_tx_list(address)
        token_txs = self._client.get_token_transfers(address)
        funder = derive_funder(address, txs)
        if funder:
            self._funder_ledger.record(funder, address)
        return compute_airdrop_features(
            address, txs, token_txs,
            lambda f: self._funder_ledger.funded_count(f, address),
        )


def derive_funder(address: str, txs: list[dict]) -> str | None:
    """The wallet's funder = the `from` of its earliest inbound value-bearing tx."""
    address = address.lower()
    inbound_with_value = [
        tx for tx in txs
        if tx.get("to", "").lower() == address and int(tx.get("value", "0")) > 0
    ]
    if not inbound_with_value:
        return None
    earliest = min(inbound_with_value, key=lambda t: int(t["timeStamp"]))
    funder = earliest.get("from", "").lower()
    return funder or None


def compute_airdrop_features(
    address: str,
    txs: list[dict],
    token_txs: list[dict],
    funder_count_lookup: Callable[[str], int],
) -> dict[str, float]:
    address = address.lower()
    now_ts = datetime.now(timezone.utc).timestamp()

    if not txs:
        return {
            "wallet_age_days": 0.0,
            "tx_count_before_first_inflow": 0.0,
            "token_type_diversity": 0.0,
            "inflow_to_outflow_hours": 0.0,
            "shared_funder_score": 0.0,
            "inter_tx_time_variance": 0.0,
            "unique_counterparty_count": 0.0,
        }

    timestamps = [int(tx["timeStamp"]) for tx in txs]
    wallet_age_days = (now_ts - min(timestamps)) / 86400

    # Earliest inbound transfer timestamp per distinct token received.
    inbound_by_token: dict[str, int] = {}
    for t in token_txs:
        if t.get("to", "").lower() != address:
            continue
        token = t.get("contractAddress", "").lower()
        if not token:
            continue
        ts = int(t["timeStamp"])
        if token not in inbound_by_token or ts < inbound_by_token[token]:
            inbound_by_token[token] = ts

    if inbound_by_token:
        first_inflow_ts = min(inbound_by_token.values())
        tx_count_before_first_inflow = float(sum(1 for ts in timestamps if ts < first_inflow_ts))
    else:
        tx_count_before_first_inflow = float(len(timestamps))

    token_type_diversity = float(len({
        tx["contractAddress"].lower() for tx in token_txs if tx.get("contractAddress")
    }))

    # For each token received, hours to its first outbound transfer afterward;
    # take the minimum across all tokens (fastest flip = most suspicious).
    flip_hours: list[float] = []
    for token, inflow_ts in inbound_by_token.items():
        outbound = [
            t for t in token_txs
            if t.get("from", "").lower() == address
            and t.get("contractAddress", "").lower() == token
            and int(t["timeStamp"]) >= inflow_ts
        ]
        if outbound:
            first_out_ts = min(int(t["timeStamp"]) for t in outbound)
            flip_hours.append((first_out_ts - inflow_ts) / 3600)
    inflow_to_outflow_hours = min(flip_hours) if flip_hours else 0.0

    funder = derive_funder(address, txs)
    shared_funder_score = math.log1p(funder_count_lookup(funder)) if funder else 0.0

    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        gaps = [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        inter_tx_time_variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    else:
        inter_tx_time_variance = 0.0

    counterparties: set[str] = set()
    for tx in txs:
        frm = tx.get("from", "").lower()
        to = tx.get("to", "").lower()
        if frm and frm != address:
            counterparties.add(frm)
        if to and to != address:
            counterparties.add(to)
    unique_counterparty_count = float(len(counterparties))

    return {
        "wallet_age_days": wallet_age_days,
        "tx_count_before_first_inflow": tx_count_before_first_inflow,
        "token_type_diversity": token_type_diversity,
        "inflow_to_outflow_hours": inflow_to_outflow_hours,
        "shared_funder_score": shared_funder_score,
        "inter_tx_time_variance": inter_tx_time_variance,
        "unique_counterparty_count": unique_counterparty_count,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_airdrop_features.py -v
```

Expected: all 19 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/feature/airdrop_features.py tests/test_airdrop_features.py
git commit -m "feat: make airdrop feature extraction event-agnostic with funder ledger"
```

---

## Task 4: `GMMWrapper` — drop `funding_address_set`, rename feature keys

**Files:**
- Modify: `src/blockchain_ai/model/gmm_wrapper.py`
- Modify: `tests/test_gmm_wrapper.py`

- [ ] **Step 1: Update the test file**

Replace the entire contents of `tests/test_gmm_wrapper.py` with:

```python
import numpy as np
import pytest
from blockchain_ai.model.gmm_wrapper import GMMWrapper

_FEATURE_COLS = [
    "wallet_age_days",
    "tx_count_before_first_inflow",
    "token_type_diversity",
    "inflow_to_outflow_hours",
    "shared_funder_score",
    "inter_tx_time_variance",
    "unique_counterparty_count",
]

_RNG = np.random.default_rng(42)

def _synthetic_data(n: int = 200) -> np.ndarray:
    """Generate 4 clusters of synthetic wallet features."""
    genuine = _RNG.normal([365, 50, 10, 720, 0, 50000, 30], [50, 20, 5, 200, 0.1, 10000, 10], (n // 4, 7))
    casual  = _RNG.normal([180, 20, 5, 48, 0, 10000, 15], [30, 10, 3, 20, 0.1, 5000, 5], (n // 4, 7))
    light   = _RNG.normal([30, 3, 2, 4, 0.3, 1000, 5], [10, 2, 1, 2, 0.2, 500, 3], (n // 4, 7))
    heavy   = _RNG.normal([10, 1, 1, 0.5, 0.9, 100, 2], [5, 1, 0.5, 0.3, 0.1, 50, 1], (n // 4, 7))
    return np.vstack([genuine, casual, light, heavy]).clip(0)


def test_fit_returns_self():
    wrapper = GMMWrapper()
    X = _synthetic_data()
    result = wrapper.fit(X, _FEATURE_COLS)
    assert result is wrapper


def test_bic_scores_has_seven_entries():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    assert len(wrapper.bic_scores) == 7
    ks = [entry["k"] for entry in wrapper.bic_scores]
    assert ks == list(range(2, 9))


def test_bic_scores_are_floats():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    for entry in wrapper.bic_scores:
        assert isinstance(entry["bic"], float)


def test_farmer_score_in_unit_interval():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    genuine_features = {
        "wallet_age_days": 400.0, "tx_count_before_first_inflow": 60.0,
        "token_type_diversity": 12.0, "inflow_to_outflow_hours": 800.0,
        "shared_funder_score": 0.0, "inter_tx_time_variance": 60000.0,
        "unique_counterparty_count": 35.0,
    }
    score = wrapper.score_wallet(genuine_features, _FEATURE_COLS)
    assert 0.0 <= score["farmer_score"] <= 1.0


def test_heavy_farmer_scores_higher_than_genuine():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    farmer_features = {
        "wallet_age_days": 5.0, "tx_count_before_first_inflow": 0.0,
        "token_type_diversity": 1.0, "inflow_to_outflow_hours": 0.2,
        "shared_funder_score": 2.0, "inter_tx_time_variance": 50.0,
        "unique_counterparty_count": 1.0,
    }
    genuine_features = {
        "wallet_age_days": 400.0, "tx_count_before_first_inflow": 60.0,
        "token_type_diversity": 12.0, "inflow_to_outflow_hours": 800.0,
        "shared_funder_score": 0.0, "inter_tx_time_variance": 60000.0,
        "unique_counterparty_count": 35.0,
    }
    farmer_score = wrapper.score_wallet(farmer_features, _FEATURE_COLS)["farmer_score"]
    genuine_score = wrapper.score_wallet(genuine_features, _FEATURE_COLS)["farmer_score"]
    assert farmer_score > genuine_score


def test_priority_tier_normal():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    wrapper._farmer_cluster_indices = []
    features = {col: 1.0 for col in _FEATURE_COLS}
    result = wrapper.score_wallet(features, _FEATURE_COLS)
    assert result["priority_tier"] == "normal"
    assert result["farmer_score"] == 0.0


def test_priority_tier_deprioritize():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    wrapper._farmer_cluster_indices = list(range(wrapper.n_components))
    features = {col: 1.0 for col in _FEATURE_COLS}
    result = wrapper.score_wallet(features, _FEATURE_COLS)
    assert result["priority_tier"] == "deprioritize"
    assert result["farmer_score"] == pytest.approx(1.0, abs=0.01)


def test_score_wallet_result_contains_bic_scores():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    features = {col: 1.0 for col in _FEATURE_COLS}
    result = wrapper.score_wallet(features, _FEATURE_COLS)
    assert "bic_scores" in result
    assert len(result["bic_scores"]) == 7


def test_farmer_score_before_fit_raises():
    wrapper = GMMWrapper()
    with pytest.raises(RuntimeError, match="fit"):
        wrapper.farmer_score(np.ones(7))
```

(`test_funding_address_set_stored_on_fit` is removed — that field no longer exists.)

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_gmm_wrapper.py -v
```

Expected: FAIL — `KeyError: 'tx_count_before_first_inflow'` (current `_identify_farmer_clusters` still looks for the old feature names, so the farmer-signal direction logic silently no-ops for renamed columns, but this specific test set is what locks in the rename — run it to confirm today's code does NOT yet satisfy the renamed-feature assumptions, e.g. `test_heavy_farmer_scores_higher_than_genuine` may fail since the farmer signal lookup won't match any column name).

- [ ] **Step 3: Update `GMMWrapper`**

In `src/blockchain_ai/model/gmm_wrapper.py:16`, remove the `funding_address_set` field from `__init__`. Replace:

```python
        self._bic_scores: list[dict] = []
        self.funding_address_set: set[str] = set()
```

with:

```python
        self._bic_scores: list[dict] = []
```

In `src/blockchain_ai/model/gmm_wrapper.py:18-38`, replace the `fit` method:

```python
    def fit(
        self,
        X: np.ndarray,
        feature_cols: list[str],
        funding_address_set: set[str] | None = None,
    ) -> "GMMWrapper":
        X = np.asarray(X, dtype=float)
        self._feature_cols = feature_cols
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._bic_scores = self._compute_bic(X_scaled)
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._gmm.fit(X_scaled)
        self._farmer_cluster_indices = self._identify_farmer_clusters()
        if funding_address_set:
            self.funding_address_set = {a.lower() for a in funding_address_set}
        return self
```

with:

```python
    def fit(self, X: np.ndarray, feature_cols: list[str]) -> "GMMWrapper":
        X = np.asarray(X, dtype=float)
        self._feature_cols = feature_cols
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._bic_scores = self._compute_bic(X_scaled)
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._gmm.fit(X_scaled)
        self._farmer_cluster_indices = self._identify_farmer_clusters()
        return self
```

In `src/blockchain_ai/model/gmm_wrapper.py:58-64`, replace the `farmer_signals` dict inside `_identify_farmer_clusters`:

```python
        farmer_signals = {
            "wallet_age_days": "low",
            "tx_count_pre_airdrop": "low",
            "claim_to_withdraw_hours": "low",
            "gas_source_shared": "high",
            "unique_counterparty_count": "low",
        }
```

with:

```python
        farmer_signals = {
            "wallet_age_days": "low",
            "tx_count_before_first_inflow": "low",
            "inflow_to_outflow_hours": "low",
            "shared_funder_score": "high",
            "unique_counterparty_count": "low",
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_gmm_wrapper.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/blockchain_ai/model/gmm_wrapper.py tests/test_gmm_wrapper.py
git commit -m "feat: drop funding_address_set from GMMWrapper, rename farmer-signal features"
```

---

## Task 5: Router test fixtures — renamed feature columns

**Files:**
- Modify: `tests/test_router_airdrop_farmer.py:7-24`

- [ ] **Step 1: Update the fixtures**

In `tests/test_router_airdrop_farmer.py`, replace lines 7-24:

```python
_FEATURE_COLS = [
    "wallet_age_days", "tx_count_pre_airdrop", "token_type_diversity",
    "claim_to_withdraw_hours", "gas_source_shared",
    "inter_tx_time_variance", "unique_counterparty_count",
]

_RESULT = {
    "farmer_score": 0.85,
    "priority_tier": "deprioritize",
    "bic_scores": [{"k": k, "bic": float(1000 - k * 10)} for k in range(2, 9)],
    "wallet_age_days": 5.0,
    "tx_count_pre_airdrop": 0.0,
    "token_type_diversity": 1.0,
    "claim_to_withdraw_hours": 0.5,
    "gas_source_shared": 1.0,
    "inter_tx_time_variance": 100.0,
    "unique_counterparty_count": 2.0,
}
```

with:

```python
_FEATURE_COLS = [
    "wallet_age_days", "tx_count_before_first_inflow", "token_type_diversity",
    "inflow_to_outflow_hours", "shared_funder_score",
    "inter_tx_time_variance", "unique_counterparty_count",
]

_RESULT = {
    "farmer_score": 0.85,
    "priority_tier": "deprioritize",
    "bic_scores": [{"k": k, "bic": float(1000 - k * 10)} for k in range(2, 9)],
    "wallet_age_days": 5.0,
    "tx_count_before_first_inflow": 0.0,
    "token_type_diversity": 1.0,
    "inflow_to_outflow_hours": 0.5,
    "shared_funder_score": 1.1,
    "inter_tx_time_variance": 100.0,
    "unique_counterparty_count": 2.0,
}
```

No other lines in this file change — the router itself doesn't reference feature names directly.

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_router_airdrop_farmer.py -v
```

Expected: all tests PASS (router behavior is unaffected by the rename).

- [ ] **Step 3: Commit**

```bash
git add tests/test_router_airdrop_farmer.py
git commit -m "test: rename airdrop router fixtures to match renamed features"
```

---

## Task 6: Config YAML

**Files:**
- Modify: `configs/airdrop-farmer.yaml`

- [ ] **Step 1: Replace the config**

Replace the entire contents of `configs/airdrop-farmer.yaml` with:

```yaml
# Airdrop Farmer Detector
# Add as many known airdrop contracts as you have to build a richer training pool.
# Serving (POST /airdrop-farmer/analyze/{address}) needs no contract address —
# it scores any wallet generically.
task: airdrop_farmer

airdrop:
  contract_addresses:
    - "0x0000000000000000000000000000000000000000"

etherscan:
  base_url: https://api.etherscan.io/v2/api
  chain_id: 1
  rate_limit_per_sec: 5
  timeout_sec: 30

ingest:
  feature_cols:
    - wallet_age_days
    - tx_count_before_first_inflow
    - token_type_diversity
    - inflow_to_outflow_hours
    - shared_funder_score
    - inter_tx_time_variance
    - unique_counterparty_count
  fill_zero_cols:
    - inflow_to_outflow_hours
    - inter_tx_time_variance

train:
  model_type: gmm
  hyperparameters:
    n_components: 4
    covariance_type: full
    random_state: 42

serve:
  model_path: models/airdrop_farmer_gmm.joblib
  db_path: data/jobs.db
```

- [ ] **Step 2: Verify the config parses cleanly**

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from blockchain_ai.config import load_config
cfg = load_config('configs/airdrop-farmer.yaml')
print('task:', cfg.task)
print('contract_addresses:', cfg.airdrop.contract_addresses)
print('feature_cols:', cfg.ingest.feature_cols)
print('OK')
"
```

Expected output:
```
task: airdrop_farmer
contract_addresses: ['0x0000000000000000000000000000000000000000']
feature_cols: ['wallet_age_days', 'tx_count_before_first_inflow', 'token_type_diversity', 'inflow_to_outflow_hours', 'shared_funder_score', 'inter_tx_time_variance', 'unique_counterparty_count']
OK
```

- [ ] **Step 3: Commit**

```bash
git add configs/airdrop-farmer.yaml
git commit -m "feat: support multiple contract_addresses and renamed features in airdrop-farmer.yaml"
```

---

## Task 7: `app.py` wiring

**Files:**
- Modify: `app.py:86-110`

- [ ] **Step 1: Replace the `airdrop_farmer` branch**

In `app.py`, replace lines 86-110:

```python
elif task == "airdrop_farmer":
    from datetime import datetime, timezone
    from blockchain_ai.feature.airdrop_features import AirdropFeatureExtractor
    from blockchain_ai.database.job_store import JobStore
    from blockchain_ai.server.router_airdrop_farmer import create_router as create_airdrop_router

    if _cfg.airdrop is None:
        raise RuntimeError("Config task=airdrop_farmer requires an 'airdrop' section.")

    _airdrop_date = datetime.fromisoformat(_cfg.airdrop.date).replace(tzinfo=timezone.utc)
    _funding_set = model.funding_address_set if model is not None else set()
    _feature_extractor = (
        AirdropFeatureExtractor(
            _etherscan_client,
            _cfg.airdrop.contract_address,
            _airdrop_date,
            _funding_set,
        )
        if _etherscan_client is not None
        else None
    )
    _job_store = JobStore(serve.db_path)
    app.include_router(
        create_airdrop_router(_job_store, model, _feature_extractor, feature_cols)
    )
```

with:

```python
elif task == "airdrop_farmer":
    from blockchain_ai.feature.airdrop_features import AirdropFeatureExtractor
    from blockchain_ai.database.job_store import JobStore
    from blockchain_ai.database.funder_ledger import FunderLedger
    from blockchain_ai.server.router_airdrop_farmer import create_router as create_airdrop_router

    if _cfg.airdrop is None:
        raise RuntimeError("Config task=airdrop_farmer requires an 'airdrop' section.")

    _funder_ledger = FunderLedger(serve.db_path)
    _feature_extractor = (
        AirdropFeatureExtractor(_etherscan_client, _funder_ledger)
        if _etherscan_client is not None
        else None
    )
    _job_store = JobStore(serve.db_path)
    app.include_router(
        create_airdrop_router(_job_store, model, _feature_extractor, feature_cols)
    )
```

- [ ] **Step 2: Verify the app starts cleanly**

```bash
CONFIG=configs/airdrop-farmer.yaml python -c "import app; print('routes:', [r.path for r in app.app.routes])"
```

Expected output includes:
```
routes: [..., '/airdrop-farmer/analyze/{address}', ...]
```

Note: the model file won't exist yet (seed step hasn't been re-run since this change), so `model` will be `None` — that's expected; the app still starts and shows the route.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: wire FunderLedger into airdrop farmer app.py, drop contract/date args"
```

---

## Task 8: Multi-contract seed workflow

**Files:**
- Modify: `src/blockchain_ai/workflow/run_airdrop_farmer_seed.py` (full rewrite)

- [ ] **Step 1: Replace the workflow script**

Replace the entire contents of `src/blockchain_ai/workflow/run_airdrop_farmer_seed.py` with:

```python
#!/usr/bin/env python3
"""
Fetch all wallets that called any of the configured airdrop contracts,
compute features, fit the GMM model, and save artifacts.

Usage:
    python src/blockchain_ai/workflow/run_airdrop_farmer_seed.py \
        --config configs/airdrop-farmer.yaml

Output:
    models/airdrop_farmer_gmm.joblib
    data/airdrop_farmer/wallet_scores.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.database.funder_ledger import FunderLedger
from blockchain_ai.feature.airdrop_features import compute_airdrop_features, derive_funder
from blockchain_ai.model.gmm_wrapper import GMMWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit airdrop farmer GMM on seed wallets")
    parser.add_argument("--config", default="configs/airdrop-farmer.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.airdrop is None:
        raise RuntimeError("Config missing 'airdrop' section")

    client = EtherscanClient.from_config(cfg.etherscan)
    contract_addresses = cfg.airdrop.contract_addresses
    feature_cols = cfg.ingest.feature_cols
    fill_zero_cols = set(cfg.ingest.fill_zero_cols)
    hp = cfg.train.hyperparameters
    ledger = FunderLedger(cfg.serve.db_path)

    caller_addresses: set[str] = set()
    for contract_address in contract_addresses:
        print(f"Fetching transactions for contract {contract_address} ...")
        contract_txs = client.get_tx_list(contract_address)
        caller_addresses |= {
            tx["from"].lower()
            for tx in contract_txs
            if tx.get("to", "").lower() == contract_address.lower()
        }
    caller_list = list(caller_addresses)
    print(f"Found {len(caller_list)} unique caller addresses across {len(contract_addresses)} contract(s).")

    if len(caller_list) < 10:
        raise RuntimeError(
            f"Only {len(caller_list)} callers found. "
            "Check contract_addresses in the config."
        )

    # Pass 1: fetch tx data per wallet and record funders into the ledger.
    wallet_data: dict[str, tuple[list, list]] = {}
    failed = 0

    for i, address in enumerate(caller_list):
        print(f"  [{i + 1}/{len(caller_list)}] {address}")
        try:
            txs = client.get_tx_list(address)
            token_txs = client.get_token_transfers(address)
            wallet_data[address] = (txs, token_txs)
            funder = derive_funder(address, txs)
            if funder:
                ledger.record(funder, address)
        except Exception as exc:
            print(f"    WARNING: Failed for {address}: {exc}")
            failed += 1

    if failed > len(caller_list) * 0.5:
        raise RuntimeError(f"Too many failures ({failed}/{len(caller_list)}). Aborting.")

    print(f"\nSuccessfully collected {len(wallet_data)} wallets ({failed} failed).")

    # Pass 2: compute features now that the ledger has every wallet's funder recorded.
    rows: list[dict] = []
    for address, (txs, token_txs) in wallet_data.items():
        features = compute_airdrop_features(
            address, txs, token_txs,
            lambda funder, addr=address: ledger.funded_count(funder, addr),
        )
        for col in fill_zero_cols:
            if features.get(col) is None or features[col] != features[col]:
                features[col] = 0.0
        rows.append({"address": address, **features})

    # Fit the model
    X = np.array([[row[col] for col in feature_cols] for row in rows], dtype=float)
    print(f"\nFitting GMM (n_components={hp.get('n_components', 4)}) on {len(rows)} wallets ...")
    wrapper = GMMWrapper(
        n_components=int(hp.get("n_components", 4)),
        covariance_type=hp.get("covariance_type", "full"),
        random_state=int(hp.get("random_state", 42)),
    )
    wrapper.fit(X, feature_cols)

    print("\nBIC scores:")
    for entry in wrapper.bic_scores:
        print(f"  k={entry['k']}: BIC={entry['bic']:.1f}")

    # Save model
    model_path = Path(cfg.serve.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, model_path)
    print(f"\nModel saved to {model_path}")

    # Score all seed wallets and write CSV
    output_path = Path("data/airdrop_farmer/wallet_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scored_rows = []
    for row in rows:
        features = {col: row[col] for col in feature_cols}
        score_result = wrapper.score_wallet(features, feature_cols)
        scored_rows.append({
            "wallet_address": row["address"],
            "farmer_score": score_result["farmer_score"],
            "priority_tier": score_result["priority_tier"],
            **features,
        })

    with open(output_path, "w", newline="") as f:
        fieldnames = ["wallet_address", "farmer_score", "priority_tier"] + feature_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    print(f"Scores written to {output_path}")

    tier_counts: dict[str, int] = {}
    for r in scored_rows:
        tier_counts[r["priority_tier"]] = tier_counts.get(r["priority_tier"], 0) + 1
    print("\nPriority tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} ({100 * count / len(scored_rows):.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports cleanly (no live API call)**

```bash
python -c "
import sys
sys.path.insert(0, 'src')
from blockchain_ai.workflow import run_airdrop_farmer_seed
print('imports OK')
"
```

Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add src/blockchain_ai/workflow/run_airdrop_farmer_seed.py
git commit -m "feat: support multiple airdrop contracts and funder ledger in seed workflow"
```

---

## Task 9: Full regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS, including every test touched in Tasks 1-8.

- [ ] **Step 2: Confirm the app still starts with the new config**

```bash
CONFIG=configs/airdrop-farmer.yaml python -c "import app; print('OK')"
```

Expected: `OK` (model will print a WARNING and load as `None` since the old `.joblib` is now incompatible — that's expected per the migration note below, not a bug).

- [ ] **Step 3: No commit needed for this task** — it's verification only. If any test fails, fix it under whichever Task introduced the regression and re-commit there.

---

## Migration Note (manual, post-merge)

The existing `models/airdrop_farmer_gmm.joblib` was fit under the old single-contract feature semantics and the now-removed `funding_address_set` field — it is incompatible with this change. After merging, re-run the seed step to regenerate it:

```bash
python src/blockchain_ai/workflow/run_airdrop_farmer_seed.py --config configs/airdrop-farmer.yaml
```

## Usage After Implementation

**Run seed step** (pulls training wallets from every contract in `airdrop.contract_addresses`):
```bash
python src/blockchain_ai/workflow/run_airdrop_farmer_seed.py --config configs/airdrop-farmer.yaml
```

**Start the API**:
```bash
CONFIG=configs/airdrop-farmer.yaml uvicorn app:app --reload
```

**Score any wallet** (no contract address needed):
```bash
curl http://localhost:8000/airdrop-farmer/analyze/0xYOUR_WALLET_ADDRESS
# Returns 202 on first call; repeat to poll for result
```

**Add a new airdrop to the training pool**: append its contract address to `airdrop.contract_addresses` in `configs/airdrop-farmer.yaml` and re-run the seed step — no code changes needed.
