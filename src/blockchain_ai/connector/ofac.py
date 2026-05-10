import csv
import io
import requests
from datetime import datetime, timezone
from blockchain_ai.config import OFACConfig
from blockchain_ai.connector.schema import AddressRecord


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
