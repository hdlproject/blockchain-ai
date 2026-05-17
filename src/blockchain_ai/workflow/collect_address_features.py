#!/usr/bin/env python3
"""
Fetch Etherscan on-chain features for all labeled addresses.
Run after collect_labels.py.

Usage:
    python src/blockchain_ai/workflow/collect_address_features.py --config configs/address-classifier.yaml
"""
import argparse
import csv
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.address_features import AddressFeatureExtractor

LABELS_PATH = "data/processed/labels/addresses.csv"
OUTPUT_PATH = "data/processed/features/address_features.csv"


def main():
    parser = argparse.ArgumentParser(description="Collect Etherscan features for labeled addresses")
    parser.add_argument("--config", default="configs/address-classifier.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")

    labels_path = Path(LABELS_PATH)
    if not labels_path.exists():
        raise FileNotFoundError(f"{LABELS_PATH} not found. Run collect_labels.py first.")

    with open(labels_path) as f:
        labeled = [(row["address"], row["label"]) for row in csv.DictReader(f)]

    print(f"Found {len(labeled)} labeled addresses. Fetching features...")

    extractor = AddressFeatureExtractor(EtherscanClient.from_config(cfg.etherscan))
    feature_cols = cfg.ingest.feature_cols
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (address, label) in enumerate(labeled):
        print(f"  [{i + 1}/{len(labeled)}] {address}")
        try:
            features = extractor.extract(address)
            row = {col: features.get(col, 0.0) for col in feature_cols}
            row["label"] = label
            rows.append(row)
        except Exception as e:
            print(f"    WARNING: Failed for {address}: {e}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_cols + ["label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
