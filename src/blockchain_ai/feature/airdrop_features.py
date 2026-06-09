from datetime import datetime, timezone
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.feature_extractor import FeatureExtractor


class AirdropFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        client: EtherscanClient,
        contract_address: str,
        airdrop_date: datetime,
        funding_address_set: set[str] | None = None,
    ):
        self._client = client
        self._contract_address = contract_address.lower()
        self._airdrop_date = airdrop_date
        self._funding_address_set = {a.lower() for a in (funding_address_set or set())}

    def extract(self, address: str) -> dict[str, float]:
        address = address.lower()
        txs = self._client.get_tx_list(address)
        token_txs = self._client.get_token_transfers(address)
        return compute_airdrop_features(
            address, txs, token_txs,
            self._contract_address, self._airdrop_date, self._funding_address_set,
        )


def compute_airdrop_features(
    address: str,
    txs: list[dict],
    token_txs: list[dict],
    contract_address: str,
    airdrop_date: datetime,
    funding_address_set: set[str],
) -> dict[str, float]:
    address = address.lower()
    contract_address = contract_address.lower()
    airdrop_ts = airdrop_date.timestamp()
    now_ts = datetime.now(timezone.utc).timestamp()

    if not txs:
        return {
            "wallet_age_days": 0.0,
            "tx_count_pre_airdrop": 0.0,
            "token_type_diversity": 0.0,
            "claim_to_withdraw_hours": 0.0,
            "gas_source_shared": 0.0,
            "inter_tx_time_variance": 0.0,
            "unique_counterparty_count": 0.0,
        }

    timestamps = [int(tx["timeStamp"]) for tx in txs]
    wallet_age_days = (now_ts - min(timestamps)) / 86400

    tx_count_pre_airdrop = float(sum(1 for ts in timestamps if ts < airdrop_ts))

    token_type_diversity = float(len({
        tx["contractAddress"].lower() for tx in token_txs if tx.get("contractAddress")
    }))

    # claim event: first tx where to == contract_address
    claim_ts = None
    for tx in txs:
        if tx.get("to", "").lower() == contract_address:
            ts = int(tx["timeStamp"])
            if claim_ts is None or ts < claim_ts:
                claim_ts = ts

    claim_to_withdraw_hours = 0.0
    if claim_ts is not None:
        outbound = [
            t for t in token_txs
            if t.get("from", "").lower() == address and int(t["timeStamp"]) >= claim_ts
        ]
        if outbound:
            first_out_ts = min(int(t["timeStamp"]) for t in outbound)
            claim_to_withdraw_hours = (first_out_ts - claim_ts) / 3600

    # gas_source_shared: funding wallet = from address of earliest inbound tx with value > 0
    funding_address = None
    inbound_with_value = [
        tx for tx in txs
        if tx.get("to", "").lower() == address and int(tx.get("value", "0")) > 0
    ]
    if inbound_with_value:
        earliest = min(inbound_with_value, key=lambda t: int(t["timeStamp"]))
        funding_address = earliest.get("from", "").lower()
    gas_source_shared = 1.0 if (funding_address and funding_address in funding_address_set) else 0.0

    # inter_tx_time_variance: population variance of gaps between consecutive tx timestamps
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
        "tx_count_pre_airdrop": tx_count_pre_airdrop,
        "token_type_diversity": token_type_diversity,
        "claim_to_withdraw_hours": claim_to_withdraw_hours,
        "gas_source_shared": gas_source_shared,
        "inter_tx_time_variance": inter_tx_time_variance,
        "unique_counterparty_count": unique_counterparty_count,
    }
