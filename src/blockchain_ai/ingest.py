import numpy as np
import pandas as pd
from pathlib import Path
from blockchain_ai.config import IngestConfig


def load_and_clean(input_path: str, output_path: str, config: IngestConfig) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(path)

    df = df[config.feature_cols].copy()
    if config.fill_zero_cols:
        df[config.fill_zero_cols] = df[config.fill_zero_cols].fillna(0.0)

    # target_col is "log_<source_col>" — derive it from the source column
    source_col = config.target_col.removeprefix("log_")
    df[config.target_col] = np.log1p(df[source_col])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
