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

from blockchain_ai.block_features import BlockFeatureExtractor
from blockchain_ai.config import load_config
from blockchain_ai.etherscan import EtherscanClient

_extractor = BlockFeatureExtractor()


def apply_target_shift(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    if len(df) < 2:
        raise RuntimeError(f"Dataset has too few rows ({len(df)}) to apply target shift")
    df = df.copy()
    df[target_col] = df[source_col].shift(-1)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def _trim_csv(output_path: str, max_rows: int) -> None:
    """Drop oldest rows if CSV exceeds max_rows, keeping the most recent."""
    df = pd.read_csv(output_path)
    if len(df) > max_rows:
        df.tail(max_rows).to_csv(output_path, index=False)
        print(f"      Trimmed to {max_rows} rows (rolling window)")


def _append_new_rows(context_rows: list[dict], new_rows: list[dict], n_appended: int, output_path: str) -> int:
    """Derive features on context+new_rows, apply target shift, append only unwritten rows to CSV."""
    combined = context_rows + new_rows
    if len(combined) < 2:
        return n_appended
    df = _extractor.derive(combined)
    df = apply_target_shift(df, source_col="base_fee_gwei", target_col="next_base_fee_gwei")
    df_new = df.iloc[len(context_rows):]
    df_to_write = df_new.iloc[n_appended:]
    if df_to_write.empty:
        return n_appended
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_to_write.to_csv(output_path, mode="a", header=not output_file.exists(), index=False)
    return n_appended + len(df_to_write)


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
    max_history_blocks = cfg.collect.max_history_blocks

    print(f"[1/3] Fetching latest block number ...")
    latest = client.get_latest_block_number()
    print(f"      Latest block: {latest}")

    context_rows: list[dict] = []
    output_file = Path(output_path)
    if output_file.exists():
        existing = pd.read_csv(output_path)
        if not existing.empty and "block_number" in existing.columns:
            start_block = int(existing["block_number"].iloc[-1]) + 1
            tail = existing[["block_number", "base_fee_per_gas", "gas_used_ratio", "timestamp"]].tail(BlockFeatureExtractor.TREND_LOOKBACK)
            context_rows = [
                {
                    "block_number": int(r["block_number"]),
                    "base_fee_per_gas": int(r["base_fee_per_gas"]),
                    "gas_used_ratio": float(r["gas_used_ratio"]),
                    "timestamp": int(r["timestamp"]),
                }
                for _, r in tail.iterrows()
            ]
            print(f"      Resuming from block {start_block} ({len(existing)} existing rows in file)")
        else:
            start_block = latest - n_blocks + 1
    else:
        start_block = latest - n_blocks + 1

    if start_block > latest:
        print(f"      Already up to date (latest block {latest}). Nothing to fetch.")
        return

    stop = False

    def _handle_sigint(sig, frame):
        nonlocal stop
        print("\n      Interrupted — will write collected data ...")
        stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    block_range = range(start_block, latest + 1)
    n_new = len(block_range)
    new_rows: list[dict] = []
    n_appended = 0
    print(f"[2/3] Fetching {n_new} new blocks ({start_block} → {latest}) ...")
    for i, block_num in enumerate(block_range):
        if stop:
            break
        try:
            row = client.get_block(block_num)
        except Exception as exc:
            warnings.warn(
                f"Block {block_num} failed after all retries: {exc} — stopping early with {len(new_rows)} new rows collected"
            )
            break
        if row:
            new_rows.append(row)
        if (i + 1) % checkpoint_every == 0:
            print(f"      Fetched {i + 1}/{n_new} new blocks ({len(new_rows)} valid) ...")
            n_appended = _append_new_rows(context_rows, new_rows, n_appended, output_path)
            print(f"      Checkpoint: appended up to {n_appended} rows to {output_path}")

    if len(new_rows) < n_new:
        warnings.warn(f"Requested {n_new} new blocks but only fetched {len(new_rows)} valid (post-EIP-1559) blocks")

    print(f"[3/3] Deriving features and writing to {output_path} ...")
    n_appended = _append_new_rows(context_rows, new_rows, n_appended, output_path)
    print(f"      Appended {n_appended} new rows to {output_path}")
    _trim_csv(output_path, max_history_blocks)


if __name__ == "__main__":
    main()
