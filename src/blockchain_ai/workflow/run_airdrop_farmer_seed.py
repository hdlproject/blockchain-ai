#!/usr/bin/env python3
"""
Fetch all wallets that called the airdrop contract, compute features,
fit the GMM model, and save artifacts.

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
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from blockchain_ai.config import load_config
from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.airdrop_features import compute_airdrop_features
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
    contract_address = cfg.airdrop.contract_address
    airdrop_date = datetime.fromisoformat(cfg.airdrop.date).replace(tzinfo=timezone.utc)
    feature_cols = cfg.ingest.feature_cols
    fill_zero_cols = set(cfg.ingest.fill_zero_cols)
    hp = cfg.train.hyperparameters

    print(f"Fetching transactions for contract {contract_address} ...")
    contract_txs = client.get_tx_list(contract_address)
    caller_addresses = list({
        tx["from"].lower()
        for tx in contract_txs
        if tx.get("to", "").lower() == contract_address.lower()
    })
    print(f"Found {len(caller_addresses)} unique caller addresses.")

    if len(caller_addresses) < 10:
        raise RuntimeError(
            f"Only {len(caller_addresses)} callers found. "
            "Check AIRDROP_CONTRACT_ADDRESS in the config."
        )

    # Pass 1: collect features (gas_source_shared=0) and store per-wallet funder
    rows: list[dict] = []
    wallet_funder: dict[str, str] = {}  # address → funding wallet address
    failed = 0

    for i, address in enumerate(caller_addresses):
        print(f"  [{i + 1}/{len(caller_addresses)}] {address}")
        try:
            txs = client.get_tx_list(address)
            token_txs = client.get_token_transfers(address)
            features = compute_airdrop_features(
                address, txs, token_txs, contract_address, airdrop_date, set()
            )
            # Store the funding address for pass 2
            inbound_with_value = [
                tx for tx in txs
                if tx.get("to", "").lower() == address and int(tx.get("value", "0")) > 0
            ]
            if inbound_with_value:
                earliest = min(inbound_with_value, key=lambda t: int(t["timeStamp"]))
                funder = earliest.get("from", "").lower()
                if funder:
                    wallet_funder[address] = funder

            for col in fill_zero_cols:
                if features.get(col) is None or features[col] != features[col]:
                    features[col] = 0.0

            rows.append({"address": address, **features})
        except Exception as exc:
            print(f"    WARNING: Failed for {address}: {exc}")
            failed += 1

    if failed > len(caller_addresses) * 0.5:
        raise RuntimeError(f"Too many failures ({failed}/{len(caller_addresses)}). Aborting.")

    print(f"\nSuccessfully collected {len(rows)} wallets ({failed} failed).")

    # Pass 2: compute gas_source_shared using the full funder→wallet map.
    # A funder is "shared" if it funded ≥2 wallets in the dataset.
    funder_count: dict[str, int] = {}
    for funder in wallet_funder.values():
        funder_count[funder] = funder_count.get(funder, 0) + 1
    shared_funders = {funder for funder, count in funder_count.items() if count >= 2}
    all_funding_addresses = set(wallet_funder.values())

    print(f"Unique funding addresses: {len(all_funding_addresses)}")
    print(f"Shared funding addresses (funded ≥2 wallets): {len(shared_funders)}")

    for row in rows:
        addr = row["address"]
        funder = wallet_funder.get(addr)
        row["gas_source_shared"] = 1.0 if (funder and funder in shared_funders) else 0.0

    # Fit the model
    X = np.array([[row[col] for col in feature_cols] for row in rows], dtype=float)
    print(f"\nFitting GMM (n_components={hp.get('n_components', 4)}) on {len(rows)} wallets ...")
    wrapper = GMMWrapper(
        n_components=int(hp.get("n_components", 4)),
        covariance_type=hp.get("covariance_type", "full"),
        random_state=int(hp.get("random_state", 42)),
    )
    wrapper.fit(X, feature_cols, funding_address_set=all_funding_addresses)

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
