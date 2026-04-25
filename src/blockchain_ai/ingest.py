import pandas as pd
from pathlib import Path


def load_and_clean(input_path: str, output_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(path)
    df = df.dropna()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
