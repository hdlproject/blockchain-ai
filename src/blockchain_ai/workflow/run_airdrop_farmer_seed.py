#!/usr/bin/env python3
"""
Fetch all wallets that called any of the configured airdrop contracts,
compute features, fit the GMM model, and save artifacts.

Usage:
    python src/blockchain_ai/workflow/run_airdrop_farmer_seed.py \
        --config configs/airdrop-farmer.yaml

Output:
    models/airdrop_farmer_gmm.joblib
    data/airdrop_farmer/wallet_scores.csv
"""
import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit airdrop farmer GMM on seed wallets")
    parser.add_argument("--config", default="configs/airdrop-farmer.yaml")
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

    caller_addresses: set[str] = set()
    for contract_address in contract_addresses:
        print(f"Fetching transactions for contract {contract_address} ...")
        contract_txs = client.get_tx_list(contract_address)
        caller_addresses |= {
            tx["from"].lower()
            for tx in contract_txs
            if tx.get("to", "").lower() == contract_address.lower()
        }
    caller_list = list(caller_addresses)
    print(f"Found {len(caller_list)} unique caller addresses across {len(contract_addresses)} contract(s).")

    if len(caller_list) < 10:
        raise RuntimeError(
            f"Only {len(caller_list)} callers found. "
            "Check contract_addresses in the config."
        )

    # Pass 1: fetch tx data per wallet and record funders into the ledger.
    wallet_data: dict[str, tuple[list, list]] = {}
    failed = 0

    for i, address in enumerate(caller_list):
        print(f"  [{i + 1}/{len(caller_list)}] {address}")
        try:
            txs = client.get_tx_list(address)
            token_txs = client.get_token_transfers(address)
            wallet_data[address] = (txs, token_txs)
            funder = derive_funder(address, txs)
            if funder:
                ledger.record(funder, address)
        except Exception as exc:
            print(f"    WARNING: Failed for {address}: {exc}")
            failed += 1

    if failed > len(caller_list) * 0.5:
        raise RuntimeError(f"Too many failures ({failed}/{len(caller_list)}). Aborting.")

    print(f"\nSuccessfully collected {len(wallet_data)} wallets ({failed} failed).")

    # Pass 2: compute features now that the ledger has every wallet's funder recorded.
    rows: list[dict] = []
    for address, (txs, token_txs) in wallet_data.items():
        features = compute_airdrop_features(
            address, txs, token_txs,
            lambda funder, addr=address: ledger.funded_count(funder, addr),
        )
        for col in fill_zero_cols:
            if features.get(col) is None or features[col] != features[col]:
                features[col] = 0.0
        rows.append({"address": address, **features})

    # Fit the model
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

    # Save model
    model_path = Path(cfg.serve.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(wrapper, model_path)
    print(f"\nModel saved to {model_path}")

    # Score all seed wallets and write CSV
    output_path = Path("data/airdrop_farmer/wallet_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scored_rows = []
    for row in rows:
        features = {col: row[col] for col in feature_cols}
        score_result = wrapper.score_wallet(features, feature_cols)
        scored_rows.append({
            "wallet_address": row["address"],
            "farmer_score": score_result["farmer_score"],
            "priority_tier": score_result["priority_tier"],
            **features,
        })

    with open(output_path, "w", newline="") as f:
        fieldnames = ["wallet_address", "farmer_score", "priority_tier"] + feature_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    print(f"Scores written to {output_path}")

    tier_counts: dict[str, int] = {}
    for r in scored_rows:
        tier_counts[r["priority_tier"]] = tier_counts.get(r["priority_tier"], 0) + 1
    print("\nPriority tier distribution:")
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count} ({100 * count / len(scored_rows):.1f}%)")


if __name__ == "__main__":
    main()
