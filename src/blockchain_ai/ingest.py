import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from blockchain_ai.config import IngestConfig


def load_and_clean(input_path: str, output_path: str, config: IngestConfig) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(csv_name))

    df = df.drop(columns=[c for c in config.drop_cols if c in df.columns])
    df[config.fill_zero_cols] = df[config.fill_zero_cols].fillna(0.0)
    df[config.timestamp_col] = (
        pd.to_datetime(df[config.timestamp_col], utc=True).astype("int64") // 10**9
    )
    df[f"log_{config.target_col}"] = np.log1p(df[config.target_col])
    df = df.drop(columns=[config.target_col])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
