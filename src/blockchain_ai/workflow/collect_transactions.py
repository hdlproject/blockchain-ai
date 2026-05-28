#!/usr/bin/env python3
"""
Sweep recent Ethereum blocks for transactions and extract features for anomaly detection.

Usage:
    poetry run python src/blockchain_ai/workflow/collect_transactions.py \
        --config configs/transaction-anomaly-detector.yaml
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.transaction_features import TransactionFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Collect Ethereum transactions for anomaly detection")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.collect is None:
        raise RuntimeError("Config missing 'collect' section")
    if cfg.paths is None:
        raise RuntimeError("Config missing 'paths' section")

    client = EtherscanClient.from_config(cfg.etherscan)
    n_blocks = cfg.collect.n_blocks
    output_path = cfg.collect.output_path
    processed_path = cfg.paths.processed_path

    print(f"[1/3] Fetching latest block number ...")
    latest = client.get_latest_block_number()
    start_block = latest - n_blocks + 1
    print(f"      Fetching blocks {start_block} → {latest} ({n_blocks} blocks)")

    all_txs: list[dict] = []
    print(f"[2/3] Fetching transactions ...")
    for i, block_num in enumerate(range(start_block, latest + 1)):
        try:
            txs = client.get_block_with_txs(block_num)
            if txs:
                all_txs.extend(txs)
        except Exception as exc:
            warnings.warn(f"Block {block_num} failed: {exc}")
        if (i + 1) % 50 == 0:
            print(f"      Fetched {i + 1}/{n_blocks} blocks ({len(all_txs)} txs so far) ...")

    if not all_txs:
        raise RuntimeError("No transactions collected — check ETHERSCAN_API_KEY and network.")

    print(f"[3/3] Extracting features from {len(all_txs)} transactions ...")
    extractor = TransactionFeatureExtractor()
    df = extractor.extract_dataset(all_txs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Saved {len(df)} raw rows to {output_path}")

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df[cfg.ingest.feature_cols].to_csv(processed_path, index=False)
    print(f"      Saved {len(df)} processed rows to {processed_path}")


if __name__ == "__main__":
    main()
