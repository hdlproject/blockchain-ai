import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class AddressRecord:
    address: str
    chain_id: int
    label: str        # sanctioned | scammer | phishing | unknown
    confidence: float
    sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        return {
            "address": self.address,
            "chain_id": self.chain_id,
            "label": self.label,
            "confidence": self.confidence,
            "sources": "|".join(self.sources),
            "flags": "|".join(self.flags),
            "fetched_at": self.fetched_at,
        }


@dataclass
class TokenRecord:
    token_address: str
    chain_id: int
    is_risky: bool
    risk_score: float
    sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_row(self) -> dict:
        return {
            "token_address": self.token_address,
            "chain_id": self.chain_id,
            "is_risky": self.is_risky,
            "risk_score": self.risk_score,
            "sources": "|".join(self.sources),
            "flags": "|".join(self.flags),
            "fetched_at": self.fetched_at,
        }


ADDRESS_FIELDNAMES = ["address", "chain_id", "label", "confidence", "sources", "flags", "fetched_at"]
TOKEN_FIELDNAMES = ["token_address", "chain_id", "is_risky", "risk_score", "sources", "flags", "fetched_at"]


def write_address_csv(records: list[AddressRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ADDRESS_FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_row())


def write_token_csv(records: list[TokenRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TOKEN_FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_row())
