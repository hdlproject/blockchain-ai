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
