#!/usr/bin/env python3
"""
Run the full ML pipeline: ingest/features -> [hpo] -> train -> evaluate.

Regression usage:
    poetry run python src/blockchain_ai/workflow/run_pipeline.py \
        --input data/raw/ethereum-blocks.csv \
        --config configs/ethereum-gas-price.yaml

Classification usage (features CSV must already exist from collect_address_features.py):
    poetry run python src/blockchain_ai/workflow/run_pipeline.py \
        --config configs/address-classifier.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from blockchain_ai.config import load_config
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline")
    parser.add_argument("--input", default=None, help="Path to raw input CSV (required for regression)")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg.paths is None:
        raise RuntimeError("Config missing required 'paths' section (processed_path, test_path, report_path).")

    processed_path = cfg.paths.processed_path
    test_path = cfg.paths.test_path
    report_path = cfg.paths.report_path
    model_path = cfg.serve.model_path if cfg.serve else "models/model.joblib"

    if cfg.task == "regression":
        if not args.input:
            parser.error("--input is required for task=regression")

        from blockchain_ai.feature.feature_engineering import load_and_clean
        print(f"[1/3] Ingesting {args.input} ...")
        load_and_clean(args.input, processed_path, cfg)
    elif cfg.task == "classification":
        if args.input:
            processed_path = args.input

        if not Path(processed_path).exists():
            raise FileNotFoundError(
                f"{processed_path} not found. Run collect_address_features.py first."
            )
        print(f"[1/3] Ingesting {args.input} ...")
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")

    print(f"[2/3] Training {cfg.train.model_type} model ...")
    train_model(processed_path, model_path, test_path, cfg)

    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(test_path, model_path, report_path, cfg)

    print(f"\nPipeline complete. Report:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
