import requests
from datetime import datetime, timezone
from blockchain_ai.config import MEWConfig
from blockchain_ai.connector.schema import AddressRecord


class MEWClient:
    def __init__(self, url: str, timeout_sec: int):
        self._url = url
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: MEWConfig) -> "MEWClient":
        return cls(url=config.url, timeout_sec=config.timeout_sec)

    def fetch_eth_addresses(self) -> list[AddressRecord]:
        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()
        # Response is a list of {"address": "0x...", "comment": "...", "date": "..."}
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for entry in response.json():
            address = entry.get("address", "").strip().lower()
            if not address.startswith("0x") or len(address) != 42:
                continue
            records.append(AddressRecord(
                address=address, chain_id=1, label="phishing", confidence=0.9,
                sources=["mew"], flags=["phishing"], fetched_at=now,
            ))
        return records
