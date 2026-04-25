# Regression Model Project Design

**Date:** 2026-04-25

## Overview

A production-ready regression model project that runs on a schedule, reads from CSV files, trains a model, evaluates it, and saves benchmark reports.

## Project Structure

```
blockchain-ai/
├── data/
│   ├── raw/          # original CSVs, never modified
│   └── processed/    # cleaned/transformed data
├── models/           # saved model artifacts (.pkl, .joblib)
├── notebooks/        # exploratory analysis
├── reports/          # benchmark/evaluation outputs
├── src/
│   └── blockchain_ai/
│       ├── __init__.py
│       ├── ingest.py      # load & validate CSV
│       ├── train.py       # train model
│       ├── evaluate.py    # metrics & benchmarks
│       └── predict.py     # run inference
├── scripts/
│   └── run_pipeline.py    # scheduled entry point
├── pyproject.toml
└── README.md
```

## Components

- **ingest.py** — loads CSV from `data/raw/`, validates schema, outputs cleaned data to `data/processed/`
- **train.py** — reads processed data, trains a regression model (pluggable: linear, XGBoost, etc.), saves artifact to `models/`
- **evaluate.py** — runs benchmarks (RMSE, MAE, R²) against the trained model, writes report to `reports/`
- **predict.py** — loads saved model artifact and runs inference on new CSV input
- **run_pipeline.py** — orchestrates ingest → train → evaluate in sequence; intended as the scheduled entry point

## Data Flow

```
data/raw/*.csv → ingest.py → data/processed/*.csv → train.py → models/*.joblib
                                                              → evaluate.py → reports/*.json
```

## Scheduling

The pipeline is triggered via `scripts/run_pipeline.py`. Scheduling mechanism (cron, Airflow, etc.) to be decided later.

## Dependencies

- `pandas` — CSV ingestion and processing
- `scikit-learn` — regression models and metrics
- `joblib` — model serialization

## Open Decisions

- Target variable and feature columns (to be specified per use case)
- Specific regression algorithm (pluggable at train time)
- Scheduling mechanism for production
