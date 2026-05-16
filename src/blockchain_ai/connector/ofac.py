import csv
import io
import re
import requests
from datetime import datetime, timezone
from blockchain_ai.config import OFACConfig
from blockchain_ai.connector.schema import AddressRecord

# ETH addresses live in the Remarks column of sdn.csv, e.g.:
# "Digital Currency Address - ETH 0xABC...; Digital Currency Address - ETH 0xDEF...;"
_ETH_PATTERN = re.compile(r'Digital Currency Address - ETH\s+(0x[0-9a-fA-F]{40})', re.IGNORECASE)


class OFACFetcher:
    def __init__(self, sdn_url: str, timeout_sec: int):
        self._sdn_url = sdn_url
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: OFACConfig) -> "OFACFetcher":
        return cls(sdn_url=config.sdn_url, timeout_sec=config.timeout_sec)

    def fetch_eth_addresses(self) -> list[AddressRecord]:
        response = requests.get(self._sdn_url, timeout=self._timeout)
        response.raise_for_status()
        # sdn.csv columns (0-indexed):
        # 0=ent_num, 1=SDN_Name, 2=SDN_Type, 3=Program, 4=Title,
        # 5=Call_Sign, 6=Vess_type, 7=Tonnage, 8=GRT, 9=Vess_flag,
        # 10=Vess_owner, 11=Remarks
        reader = csv.reader(io.StringIO(response.text))
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for row in reader:
            if len(row) < 12:
                continue
            for match in _ETH_PATTERN.finditer(row[11]):
                records.append(AddressRecord(
                    address=match.group(1).lower(), chain_id=1, label="sanctioned", confidence=1.0,
                    sources=["ofac"], flags=["ofac_sdn"], fetched_at=now,
                ))
        return records
