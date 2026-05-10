import time
import requests
from datetime import datetime, timezone
from blockchain_ai.config import GoPlusConfig
from blockchain_ai.connector.schema import AddressRecord, TokenRecord

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
                flags = [f.removeprefix("is_") for f in _TOKEN_RISK_FLAGS if data.get(f) == "1"]
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
