import math
from datetime import datetime, timezone
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.feature_extractor import FeatureExtractor


class AddressFeatureExtractor(FeatureExtractor):
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
