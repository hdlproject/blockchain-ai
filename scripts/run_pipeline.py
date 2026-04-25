#!/usr/bin/env python3
"""
Run the full regression pipeline: ingest -> train -> evaluate.

Usage:
    poetry run python scripts/run_pipeline.py \
        --raw data/raw/input.csv \
        --target target \
        [--model-type linear]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from blockchain_ai.ingest import load_and_clean
from blockchain_ai.train import train_model
from blockchain_ai.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Run regression pipeline")
    parser.add_argument("--raw", required=True, help="Path to raw input CSV")
    parser.add_argument("--target", required=True, help="Name of target column")
    parser.add_argument("--model-type", default="linear", help="Model type (default: linear)")
    args = parser.parse_args()

    processed_path = "data/processed/processed.csv"
    model_path = "models/model.joblib"
    report_path = "reports/report.json"

    print(f"[1/3] Ingesting {args.raw} ...")
    load_and_clean(args.raw, processed_path)

    print(f"[2/3] Training {args.model_type} model ...")
    train_model(processed_path, args.target, model_path, model_type=args.model_type)

    print(f"[3/3] Evaluating model ...")
    report = evaluate_model(processed_path, args.target, model_path, report_path)

    print(f"\nPipeline complete. Report saved to {report_path}:")
    for k, v in report.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
