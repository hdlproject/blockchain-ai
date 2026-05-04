#!/usr/bin/env python3
"""
Run the full ML pipeline: ingest/features -> [hpo] -> train -> evaluate.

Regression usage:
    poetry run python scripts/run_pipeline.py \
        --input data/raw/ethereum-blocks.csv \
        --config configs/ethereum-gas-price.yaml

Classification usage (features CSV must already exist from collect_address_features.py):
    poetry run python scripts/run_pipeline.py \
        --config configs/address-classifier.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.config import load_config
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--input", default=None, help="Path to raw input CSV (required for regression)")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg.task

    if task == "regression":
        if not args.input:
            parser.error("--input is required for task=regression")
        from blockchain_ai.ingest import load_and_clean
        from blockchain_ai.tune import run_hpo

        processed_path = "data/processed/ethereum-transactions.csv"
        test_path = "data/processed/ethereum-transactions-test.csv"
        model_path = "models/model.joblib"
        report_path = "reports/report.json"

        print(f"[1/4] Ingesting {args.input} ...")
        load_and_clean(args.input, processed_path, cfg.ingest)

        train_config = cfg.train
        if cfg.hpo is not None:
            print(f"[2/4] Running Optuna HPO ({cfg.hpo.n_trials} trials) ...")
            train_config = run_hpo(processed_path, cfg.train, n_trials=cfg.hpo.n_trials)
            print(f"      Best hyperparameters: {train_config.hyperparameters}")
        else:
            print("[2/4] Skipping HPO (no hpo section in config)")

        print(f"[3/4] Training {train_config.model_type} model ...")
        train_model(processed_path, model_path, test_path, train_config)

        print("[4/4] Evaluating model ...")
        report = evaluate_model(test_path, train_config.target_col, model_path, report_path)

    else:
        features_path = "data/processed/features/address_features.csv"
        model_path = cfg.serve.model_path if cfg.serve else "models/address_classifier.joblib"
        test_path = "data/processed/address_features_test.csv"
        report_path = "reports/address_classifier_report.json"

        if not Path(features_path).exists():
            raise FileNotFoundError(f"{features_path} not found. Run collect_address_features.py first.")

        print(f"[1/2] Training classification model from {features_path} ...")
        train_model(features_path, model_path, test_path, cfg.train, task="classification")

        print("[2/2] Evaluating model ...")
        report = evaluate_model(test_path, cfg.train.target_col, model_path, report_path, task="classification")

    print(f"\nPipeline complete. Report:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
