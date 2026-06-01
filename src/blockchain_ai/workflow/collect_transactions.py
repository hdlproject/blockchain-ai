#!/usr/bin/env python3
"""
Collect Ethereum transactions sampled across hourly slots for anomaly detection.

For each hour slot in the last N days, fetches one block and randomly samples
up to --txs-per-hour transactions. This ensures hour_of_day covers all 24 hours,
giving the DBSCAN scaler a realistic feature distribution.

Usage:
    poetry run python src/blockchain_ai/workflow/collect_transactions.py \
        --config configs/transaction-anomaly-detector.yaml

    # Override defaults:
    poetry run python src/blockchain_ai/workflow/collect_transactions.py \
        --config configs/transaction-anomaly-detector.yaml \
        --days 3 --txs-per-hour 30
"""
import argparse
import random
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.transaction_features import TransactionFeatureExtractor


def _hour_slots(days: int) -> list[int]:
    now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [int((now - timedelta(hours=h + 1)).timestamp()) for h in range(days * 24)]


def main():
    parser = argparse.ArgumentParser(description="Collect Ethereum transactions for anomaly detection")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument("--days", type=int, default=None, help="Past days to cover (overrides config)")
    parser.add_argument("--txs-per-hour", type=int, dest="txs_per_hour", default=None,
                        help="Max transactions per hour slot (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.collect is None:
        raise RuntimeError("Config missing 'collect' section")
    if cfg.paths is None:
        raise RuntimeError("Config missing 'paths' section")

    days = args.days if args.days is not None else cfg.collect.days
    txs_per_hour = args.txs_per_hour if args.txs_per_hour is not None else cfg.collect.txs_per_hour
    output_path = cfg.collect.output_path
    processed_path = cfg.paths.processed_path

    slots = _hour_slots(days)
    print(f"Collecting up to {txs_per_hour} txs/hour across {days} days ({len(slots)} hour slots)")
    print(f"Estimated API calls: {len(slots) * 2}  (~{len(slots) * 2 // 5}s at 5 req/s)\n")

    client = EtherscanClient.from_config(cfg.etherscan)
    all_txs: list[dict] = []
    failed = 0

    for i, ts in enumerate(slots):
        hour_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:00 UTC")
        try:
            block_num = client.get_block_number_by_timestamp(ts)
            txs = client.get_block_with_txs(block_num)
            if txs:
                sampled = random.sample(txs, min(txs_per_hour, len(txs)))
                all_txs.extend(sampled)
        except Exception as exc:
            warnings.warn(f"Slot {hour_str} failed: {exc}")
            failed += 1

        if (i + 1) % 24 == 0 or (i + 1) == len(slots):
            pct = (i + 1) / len(slots) * 100
            print(f"  [{i + 1:>4}/{len(slots)}] {pct:5.1f}%  —  {len(all_txs)} txs collected  ({failed} slots failed)")

    if not all_txs:
        raise RuntimeError("No transactions collected — check ETHERSCAN_API_KEY and network.")

    print(f"\nExtracting features from {len(all_txs)} transactions ...")
    df = TransactionFeatureExtractor().extract_dataset(all_txs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} raw rows → {output_path}")

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    df[cfg.ingest.feature_cols].to_csv(processed_path, index=False)
    print(f"Saved {len(df)} processed rows → {processed_path}")


if __name__ == "__main__":
    main()
