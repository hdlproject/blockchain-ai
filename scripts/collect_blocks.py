#!/usr/bin/env python3
"""
Fetch the latest N blocks from Etherscan and write a feature CSV for training.

Usage:
    poetry run python scripts/collect_blocks.py --config configs/ethereum-gas-price.yaml
"""
import argparse
import signal
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


def _write_csv(rows: list[dict], output_path: str) -> int:
    df = derive_features(rows)
    df = apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return len(df)


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
    checkpoint_every = cfg.collect.checkpoint_every

    print(f"[1/3] Fetching latest block number ...")
    latest = client.get_latest_block_number()
    print(f"      Latest block: {latest}")

    stop = False

    def _handle_sigint(sig, frame):
        nonlocal stop
        print("\n      Interrupted — will write collected data ...")
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"[2/3] Fetching fee history for {n_blocks} blocks ...")
    rows: list[dict] = []
    for i, block_num in enumerate(range(latest - n_blocks + 1, latest + 1)):
        if stop:
            break
        try:
            row = client.get_block(block_num)
        except Exception as exc:
            warnings.warn(
                f"Block {block_num} failed after all retries: {exc} — stopping early with {len(rows)} rows collected"
            )
            break
        if row:
            rows.append(row)
        if (i + 1) % checkpoint_every == 0:
            print(f"      Fetched {i + 1}/{n_blocks} blocks ({len(rows)} valid) ...")
            if len(rows) >= 2:
                _write_csv(rows, output_path)
                print(f"      Checkpoint: wrote {len(rows)} rows to {output_path}")

    if len(rows) < n_blocks:
        warnings.warn(f"Requested {n_blocks} blocks but only got {len(rows)} valid (post-EIP-1559) blocks")

    print(f"[3/3] Deriving features and writing to {output_path} ...")
    n_written = _write_csv(rows, output_path)
    print(f"      Wrote {n_written} rows to {output_path}")


if __name__ == "__main__":
    main()
