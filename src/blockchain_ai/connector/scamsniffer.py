import requests
from datetime import datetime, timezone
from blockchain_ai.config import ScamSnifferConfig
from blockchain_ai.connector.schema import AddressRecord


class ScamSnifferClient:
    def __init__(self, url: str, timeout_sec: int):
        self._url = url
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: ScamSnifferConfig) -> "ScamSnifferClient":
        return cls(url=config.url, timeout_sec=config.timeout_sec)

    def fetch_eth_addresses(self) -> list[AddressRecord]:
        response = requests.get(self._url, timeout=self._timeout)
        response.raise_for_status()
        # Response is a plain list of lowercase hex address strings
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for address in response.json():
            address = address.strip().lower()
            if not address.startswith("0x") or len(address) != 42:
                continue
            records.append(AddressRecord(
                address=address, chain_id=1, label="phishing", confidence=0.9,
                sources=["scamsniffer"], flags=["phishing"], fetched_at=now,
            ))
        return records
