import csv
from pathlib import Path
from blockchain_ai.connector.schema import AddressRecord, write_address_csv


def unify_addresses(raw_paths: list[Path], output_path: Path) -> list[AddressRecord]:
    by_address: dict[str, AddressRecord] = {}
    for path in raw_paths:
        if not Path(path).exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                addr = row["address"].lower()
                sources = [s for s in row["sources"].split("|") if s]
                flags = [fl for fl in row["flags"].split("|") if fl]
                confidence = float(row["confidence"])
                if addr not in by_address:
                    by_address[addr] = AddressRecord(
                        address=addr, chain_id=int(row["chain_id"]),
                        label=row["label"], confidence=confidence,
                        sources=sources, flags=flags, fetched_at=row["fetched_at"],
                    )
                else:
                    existing = by_address[addr]
                    existing.sources = list(set(existing.sources + sources))
                    existing.flags = list(set(existing.flags + flags))
                    if confidence > existing.confidence:
                        existing.label = row["label"]
                        existing.confidence = confidence
    records = list(by_address.values())
    write_address_csv(records, output_path)
    return records
