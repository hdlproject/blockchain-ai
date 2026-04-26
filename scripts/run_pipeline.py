#!/usr/bin/env python3
"""
Run the full regression pipeline: ingest -> [hpo] -> train -> evaluate.

Usage:
    poetry run python scripts/run_pipeline.py \
        --raw data/raw/ethereum-transactions.zip \
        --config configs/ethereum-gas-price.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.ingest import load_and_clean
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model
from blockchain_ai.tune import run_hpo


def main():
    parser = argparse.ArgumentParser(description="Run regression pipeline")
    parser.add_argument("--raw", required=True, help="Path to raw input zip")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_path = "data/processed/ethereum-transactions.csv"
    test_path = "data/processed/ethereum-transactions-test.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/4] Ingesting {args.raw} ...")
    load_and_clean(args.raw, processed_path, cfg.ingest)

    train_config = cfg.train
    if cfg.hpo is not None:
        print(f"[2/4] Running Optuna HPO ({cfg.hpo.n_trials} trials) ...")
        train_config = run_hpo(processed_path, cfg.train, n_trials=cfg.hpo.n_trials)
        print(f"      Best hyperparameters: {train_config.hyperparameters}")
    else:
        print(f"[2/4] Skipping HPO (no hpo section in config)")

    print(f"[3/4] Training {train_config.model_type} model ...")
    train_model(processed_path, model_path, test_path, train_config)

    print(f"[4/4] Evaluating model ...")
    report = evaluate_model(test_path, train_config.target_col, model_path, report_path)

    print(f"\nPipeline complete. Report saved to {report_path}:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
