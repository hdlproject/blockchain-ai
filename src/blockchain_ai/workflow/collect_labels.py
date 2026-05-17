#!/usr/bin/env python3
"""
Collect ground-truth address labels from GoPlus, OFAC, Scam Sniffer, and MEW, then unify them.

Usage:
    python src/blockchain_ai/workflow/collect_labels.py --config configs/address-classifier.yaml
    python src/blockchain_ai/workflow/collect_labels.py --config configs/address-classifier.yaml \
        --addresses 0xABC...,0xDEF...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from blockchain_ai.config import load_config
from blockchain_ai.connector.goplus import GoPlusClient
from blockchain_ai.connector.mew import MEWClient
from blockchain_ai.connector.ofac import OFACFetcher
from blockchain_ai.connector.scamsniffer import ScamSnifferClient
from blockchain_ai.connector.schema import write_address_csv
from blockchain_ai.connector.unify import unify_addresses


def main():
    parser = argparse.ArgumentParser(description="Collect address labels from GoPlus, OFAC, Scam Sniffer, and MEW")
    parser.add_argument("--config", default="configs/address-classifier.yaml")
    parser.add_argument("--addresses", default="", help="Comma-separated addresses for GoPlus address security")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path("data/raw/labels")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data/processed/labels")
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = []

    if args.addresses and cfg.goplus:
        print("[goplus] Fetching address security...")
        client = GoPlusClient.from_config(cfg.goplus)
        addr_list = [a.strip() for a in args.addresses.split(",") if a.strip()]
        records = [r for a in addr_list if (r := client.get_address_security(a)) is not None]
        path = raw_dir / "goplus_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    if cfg.ofac:
        print("[ofac] Fetching sanctioned ETH addresses...")
        records = OFACFetcher.from_config(cfg.ofac).fetch_eth_addresses()
        path = raw_dir / "ofac_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    if cfg.scamsniffer:
        print("[scamsniffer] Fetching phishing addresses...")
        records = ScamSnifferClient.from_config(cfg.scamsniffer).fetch_eth_addresses()
        path = raw_dir / "scamsniffer_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    if cfg.mew:
        print("[mew] Fetching phishing addresses...")
        records = MEWClient.from_config(cfg.mew).fetch_eth_addresses()
        path = raw_dir / "mew_addresses.csv"
        write_address_csv(records, path)
        raw_files.append(path)
        print(f"  Saved {len(records)} records to {path}")

    print("[unify] Merging and deduplicating...")
    unified = unify_addresses(raw_files, processed_dir / "addresses.csv")
    print(f"  Unified {len(unified)} unique addresses → {processed_dir / 'addresses.csv'}")


if __name__ == "__main__":
    main()
