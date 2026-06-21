#!/usr/bin/env python3
"""
Fetch all wallets that called any of the configured airdrop contracts,
compute features, fit the GMM model, and save artifacts.

Usage:
    python src/blockchain_ai/workflow/run_airdrop_farmer_detector_seed.py \
        --config configs/airdrop-farmer-detector.yaml

Output:
    models/airdrop_farmer_detector_gmm.joblib
    data/airdrop_farmer_detector/wallet_scores.csv

Checkpointing:
    Raw tx data is written to data/airdrop_farmer_detector/.wallet_checkpoint.jsonl
    after each wallet fetch. On restart, already-fetched wallets are skipped.
    The checkpoint is removed on successful completion.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.database.funder_ledger import FunderLedger
from blockchain_ai.feature.airdrop_features import compute_airdrop_features, derive_funder
from blockchain_ai.model.gmm_wrapper import GMMWrapper

_OUTPUT_DIR = Path("data/airdrop_farmer_detector")
_CHECKPOINT_PATH = _OUTPUT_DIR / ".wallet_checkpoint.jsonl"
_FEATURES_PATH = _OUTPUT_DIR / "features.csv"
_SCORES_PATH = _OUTPUT_DIR / "wallet_scores.csv"


def _load_checkpoint() -> dict[str, tuple[list, list]]:
    """Return {address: (txs, token_txs)} from a previous partial run."""
    if not _CHECKPOINT_PATH.exists():
        return {}
    wallet_data: dict[str, tuple[list, list]] = {}
    with open(_CHECKPOINT_PATH) as f:
        for line in f:
            entry = json.loads(line)
            wallet_data[entry["address"]] = (entry["txs"], entry["token_txs"])
    return wallet_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit airdrop farmer detector GMM on seed wallets")
    parser.add_argument("--config", default="configs/airdrop-farmer-detector.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.etherscan is None:
        raise RuntimeError("Config missing 'etherscan' section")
    if cfg.airdrop is None:
        raise RuntimeError("Config missing 'airdrop' section")

    client = EtherscanClient.from_config(cfg.etherscan)
    contract_addresses = cfg.airdrop.contract_addresses
    feature_cols = cfg.ingest.feature_cols
    fill_zero_cols = set(cfg.ingest.fill_zero_cols)
    hp = cfg.train.hyperparameters
    ledger = FunderLedger(cfg.serve.db_path)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect caller addresses ─────────────────────────────────────────────
    max_per_contract = cfg.airdrop.max_wallets_per_contract
    caller_addresses: set[str] = set()
    for contract_address in contract_addresses:
        print(f"Fetching transactions for contract {contract_address} ...")
        contract_txs = client.get_tx_list(contract_address)
        contract_callers = [
            tx["from"].lower()
            for tx in contract_txs
            if tx.get("to", "").lower() == contract_address.lower()
        ]
        if max_per_contract is not None:
            contract_callers = contract_callers[:max_per_contract]
            print(f"  Capped to {len(contract_callers)} wallets (max_wallets_per_contract={max_per_contract}).")
        caller_addresses |= set(contract_callers)
    caller_list = list(caller_addresses)
    print(f"Found {len(caller_list)} unique caller addresses across {len(contract_addresses)} contract(s).")

    if len(caller_list) < 10:
        raise RuntimeError(
            f"Only {len(caller_list)} callers found. "
            "Check contract_addresses in the config."
        )

    # ── Pass 1: fetch tx data, checkpoint immediately, record funders ────────
    wallet_data = _load_checkpoint()
    if wallet_data:
        print(f"Resuming from checkpoint: {len(wallet_data)} wallets already fetched.")

    # Re-record funders for checkpointed wallets in case the ledger was cleared.
    for address, (txs, _) in wallet_data.items():
        funder = derive_funder(address, txs)
        if funder:
            ledger.record(funder, address)

    failed = 0
    with open(_CHECKPOINT_PATH, "a") as checkpoint_file:
        for i, address in enumerate(caller_list):
            if address in wallet_data:
                print(f"  [{i + 1}/{len(caller_list)}] {address} (cached)")
                continue
            print(f"  [{i + 1}/{len(caller_list)}] {address}")
            try:
                txs = client.get_tx_list(address)
                token_txs = client.get_token_transfers(address)
                wallet_data[address] = (txs, token_txs)
                funder = derive_funder(address, txs)
                if funder:
                    ledger.record(funder, address)
                checkpoint_file.write(
                    json.dumps({"address": address, "txs": txs, "token_txs": token_txs}) + "\n"
                )
                checkpoint_file.flush()
            except Exception as exc:
                print(f"    WARNING: Failed for {address}: {exc}")
                failed += 1

    if failed > len(caller_list) * 0.5:
        raise RuntimeError(f"Too many failures ({failed}/{len(caller_list)}). Aborting.")

    print(f"\nSuccessfully collected {len(wallet_data)} wallets ({failed} failed).")

    # ── Pass 2: compute features, write to CSV immediately ──────────────────
    print(f"\nComputing features → {_FEATURES_PATH}")
    feature_fieldnames = ["address"] + feature_cols
    rows: list[dict] = []
    with open(_FEATURES_PATH, "w", newline="") as feat_file:
        feat_writer = csv.DictWriter(feat_file, fieldnames=feature_fieldnames)
        feat_writer.writeheader()
        for address, (txs, token_txs) in wallet_data.items():
            features = compute_airdrop_features(
                address, txs, token_txs,
                lambda funder, addr=address: ledger.funded_count(funder, addr),
            )
            for col in fill_zero_cols:
                if features.get(col) is None or features[col] != features[col]:
                    features[col] = 0.0
            row = {"address": address, **features}
            rows.append(row)
            feat_writer.writerow(row)
            feat_file.flush()

    # ── Fit model ────────────────────────────────────────────────────────────
    X = np.array([[row[col] for col in feature_cols] for row in rows], dtype=float)
    print(f"\nFitting GMM (n_components={hp.get('n_components', 4)}) on {len(rows)} wallets ...")
    wrapper = GMMWrapper(
        n_components=int(hp.get("n_components", 4)),
        covariance_type=hp.get("covariance_type", "full"),
        random_state=int(hp.get("random_state", 42)),
    )
    wrapper.fit(X, feature_cols)

    print("\nBIC scores:")
    for entry in wrapper.bic_scores:
        print(f"  k={entry['k']}: BIC={entry['bic']:.1f}")

    model_path = Path(cfg.serve.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, model_path)
    print(f"\nModel saved to {model_path}")

    # ── Score wallets, write each row immediately ────────────────────────────
    print(f"\nScoring wallets → {_SCORES_PATH}")
    score_fieldnames = ["wallet_address", "farmer_score", "priority_tier"] + feature_cols
    tier_counts: dict[str, int] = {}
    with open(_SCORES_PATH, "w", newline="") as score_file:
        score_writer = csv.DictWriter(score_file, fieldnames=score_fieldnames)
        score_writer.writeheader()
        for row in rows:
            features = {col: row[col] for col in feature_cols}
            result = wrapper.score_wallet(features, feature_cols)
            tier = result["priority_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            score_writer.writerow({
                "wallet_address": row["address"],
                "farmer_score": result["farmer_score"],
                "priority_tier": tier,
                **features,
            })
            score_file.flush()

    print(f"Scores written to {_SCORES_PATH}")

    # ── Clean up checkpoint on success ───────────────────────────────────────
    _CHECKPOINT_PATH.unlink(missing_ok=True)
    print("Checkpoint cleared.")

    print("\nPriority tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} ({100 * count / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
