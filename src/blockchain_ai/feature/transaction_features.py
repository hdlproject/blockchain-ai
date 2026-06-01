import math
from datetime import datetime, timezone
import pandas as pd

from blockchain_ai.feature.feature_extractor import FeatureExtractor

# Top Ethereum tokens: address (lowercase) -> decimals
KNOWN_TOKEN_DECIMALS: dict[str, int] = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": 6,   # USDT
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,  # WETH
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,   # WBTC
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": 18,  # UNI
    "0x514910771af9ca656af840dff83e8264ecf986ca": 18,  # LINK
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": 18,  # AAVE
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": 18,  # MKR
    "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce": 18,  # SHIB
    "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2": 18,  # SUSHI
    "0xd533a949740bb3306d119cc777fa900ba034cd52": 18,  # CRV
    "0x4fabb145d64652a948d72533023f6e7a623c7c53": 18,  # BUSD
    "0xba100000625a3754423978a60c9317c58a424e3d": 18,  # BAL
    "0x0d8775f648430679a709e98d2b0cb6250d2887ef": 18,  # BAT
    "0xc00e94cb662c3520282e6f5717214004a7f26888": 18,  # COMP
    "0x3432b6a60d23ca0dfca7761b7ab56459d9c964d0": 18,  # FXS
    "0x853d955acef822db058eb8505911ed77f175b99e": 18,  # FRAX
}

_SELECTOR_TYPE: dict[str, float] = {
    "a9059cbb": 1.0,  # transfer(address,uint256)
    "23b872dd": 1.0,  # transferFrom(address,address,uint256)
    "095ea7b3": 2.0,  # approve(address,uint256)
}


def _tx_type(input_hex: str) -> float:
    if not input_hex or input_hex == "0x":
        return 0.0
    selector = input_hex[2:10].lower()
    return _SELECTOR_TYPE.get(selector, 3.0)


def _decode_erc20_amount(input_hex: str) -> int | None:
    data = input_hex[2:]
    selector = data[:8].lower()
    if selector == "a9059cbb" and len(data) >= 136:      # transfer: to(32) + amount(32)
        return int(data[72:136], 16)
    if selector == "23b872dd" and len(data) >= 200:      # transferFrom: from(32) + to(32) + amount(32)
        return int(data[136:200], 16)
    return None


def _log_transfer_value(value_eth: float, input_hex: str, to_addr: str) -> float:
    if value_eth > 0:
        return math.log1p(value_eth)
    decimals = KNOWN_TOKEN_DECIMALS.get(to_addr)
    if decimals is not None:
        raw = _decode_erc20_amount(input_hex)
        if raw and raw > 0:
            return math.log1p(raw / 10 ** decimals)
    return 0.0


class TransactionFeatureExtractor(FeatureExtractor):
    def extract(self, raw_txs: list[dict]) -> pd.DataFrame:
        return self.extract_dataset(raw_txs)

    def extract_dataset(self, raw_txs: list[dict]) -> pd.DataFrame:
        sender_counts: dict[str, int] = {}
        receiver_counts: dict[str, int] = {}

        for tx in raw_txs:
            frm = tx.get("from", "").lower()
            to = tx.get("to", "").lower()
            if frm:
                sender_counts[frm] = sender_counts.get(frm, 0) + 1
            if to:
                receiver_counts[to] = receiver_counts.get(to, 0) + 1

        rows = [
            row for tx in raw_txs
            if (row := self._extract_one(tx, sender_counts, receiver_counts)) is not None
        ]
        return pd.DataFrame(rows)

    def _extract_one(
        self,
        tx: dict,
        sender_counts: dict[str, int],
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

        if is_contract_call == 0.0:
            contract_type = 0.0  # ETH transfer
        elif to in KNOWN_TOKEN_DECIMALS:
            contract_type = 1.0  # known token
        else:
            contract_type = 2.0  # unknown contract

        return {
            "gas_price_gwei": round(gas_price_gwei, 4),
            "gas_used": gas_used,
            "input_data_len": input_data_len,
            "hour_of_day": hour_of_day,
            "sender_tx_count_window": float(sender_counts.get(frm, 1)),
            "receiver_tx_count_window": float(receiver_counts.get(to, 1)),
            "tx_type": _tx_type(input_data),
            "contract_type": contract_type,
            "log_transfer_value": round(_log_transfer_value(value_eth, input_data, to), 6),
        }
