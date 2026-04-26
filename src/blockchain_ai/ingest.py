import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

_DROP_COLS = [
    "hash", "block_hash", "from_address", "to_address", "input",
    "receipt_contract_address", "receipt_root",
]
_EIP1559_COLS = ["max_fee_per_gas", "max_priority_fee_per_gas"]


def load_and_clean(input_path: str, output_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(csv_name))

    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    df[_EIP1559_COLS] = df[_EIP1559_COLS].fillna(0.0)
    df["block_timestamp"] = (
        pd.to_datetime(df["block_timestamp"], utc=True).astype("int64") // 10**9
    )
    df["log_gas_price"] = np.log1p(df["gas_price"])
    df = df.drop(columns=["gas_price"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
