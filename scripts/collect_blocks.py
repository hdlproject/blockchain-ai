#!/usr/bin/env python3
"""
Fetch the latest N blocks from Etherscan and write a feature CSV for training.

Usage:
    poetry run python scripts/collect_blocks.py --config configs/ethereum-gas-price.yaml
"""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.etherscan import EtherscanClient


def derive_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["base_fee_gwei"] = df["base_fee_per_gas"] / 1e9
    df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.dayofweek
    lookback = 10
    shifted = df["base_fee_gwei"].shift(lookback)
    df["base_fee_trend"] = ((df["base_fee_gwei"] - shifted) / shifted).fillna(0.0)
    return df


def apply_target_shift(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    if len(df) < 2:
        raise RuntimeError(f"Dataset has too few rows ({len(df)}) to apply target shift")
    df = df.copy()
    df[target_col] = df[source_col].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect Ethereum blocks from Etherscan")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.collect is None:
        raise RuntimeError("Config missing 'collect' section")

    client = EtherscanClient.from_config(cfg.etherscan)
    n_blocks = cfg.collect.n_blocks
    output_path = cfg.collect.output_path

    print(f"[1/3] Fetching latest block number ...")
    latest = client.get_latest_block_number()
    print(f"      Latest block: {latest}")

    print(f"[2/3] Fetching fee history for {n_blocks} blocks ...")
    rows: list[dict] = []
    remaining = n_blocks
    newest = latest

    while remaining > 0:
        batch = min(remaining, 1024)
        batch_rows = client.get_fee_history(block_count=batch, newest_block=newest)
        if not batch_rows:
            warnings.warn(f"No rows returned for newest_block={newest}")
            break
        rows = batch_rows + rows
        newest = batch_rows[0]["block_number"] - 1
        remaining -= batch
        print(f"      Fetched {len(rows)} blocks so far ...")

    if len(rows) < n_blocks:
        warnings.warn(f"Requested {n_blocks} blocks but only got {len(rows)}")

    print(f"[3/3] Deriving features and writing to {output_path} ...")
    df = derive_features(rows)
    df = apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"      Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
