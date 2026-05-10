import numpy as np
import pandas as pd
from pathlib import Path
from blockchain_ai.config import PipelineConfig


def load_and_clean(input_path: str, output_path: str, cfg: PipelineConfig) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log_transform = cfg.serve.log_transform if cfg.serve else False
    config = cfg.ingest

    df = pd.read_csv(path)

    if log_transform:
        source_col = config.target_col.removeprefix("log_")
        target_series = np.log1p(df[source_col])
    else:
        target_series = df[config.target_col]

    df = df[config.feature_cols].copy()
    if config.fill_zero_cols:
        df[config.fill_zero_cols] = df[config.fill_zero_cols].fillna(0.0)

    df[config.target_col] = target_series.values

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
