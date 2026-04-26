import zipfile
import numpy as np
import pandas as pd
import pytest
from blockchain_ai.config import IngestConfig
from blockchain_ai.ingest import load_and_clean


def _make_zip(tmp_path, df):
    csv_path = tmp_path / "eth_transactions.csv"
    df.to_csv(csv_path, index=False)
    zip_path = tmp_path / "ethereum-transactions.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(csv_path, "eth_transactions.csv")
    return str(zip_path)


def _minimal_df(**overrides):
    base = {
        "hash": ["0xabc"],
        "nonce": [1],
        "transaction_index": [0],
        "from_address": ["0x1"],
        "to_address": ["0x2"],
        "value": [1000.0],
        "gas": [21000],
        "gas_price": [12000000000],
        "input": ["0x"],
        "receipt_cumulative_gas_used": [21000],
        "receipt_gas_used": [21000],
        "receipt_contract_address": [None],
        "receipt_root": [None],
        "receipt_status": [1],
        "block_timestamp": ["2023-01-01 00:00:00+00:00"],
        "block_number": [16000000],
        "block_hash": ["0xdef"],
        "max_fee_per_gas": [None],
        "max_priority_fee_per_gas": [None],
        "transaction_type": [0],
        "receipt_effective_gas_price": [12000000000],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _default_config():
    return IngestConfig(
        drop_cols=["hash", "block_hash", "from_address", "to_address", "input",
                   "receipt_contract_address", "receipt_root"],
        fill_zero_cols=["max_fee_per_gas", "max_priority_fee_per_gas"],
        timestamp_col="block_timestamp",
        target_col="gas_price",
    )


def test_load_and_clean_drops_junk_columns(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    junk = {"hash", "block_hash", "from_address", "to_address", "input",
            "receipt_contract_address", "receipt_root", "gas_price"}
    assert not junk.intersection(result.columns)


def test_load_and_clean_fills_zero_cols(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    assert result["max_fee_per_gas"].iloc[0] == 0.0
    assert result["max_priority_fee_per_gas"].iloc[0] == 0.0


def test_load_and_clean_parses_block_timestamp(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    assert pd.api.types.is_integer_dtype(result["block_timestamp"])


def test_load_and_clean_adds_log_target(tmp_path):
    df = _minimal_df(gas_price=[12000000000])
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    result = load_and_clean(zip_path, out_path, _default_config())

    assert "log_gas_price" in result.columns
    expected = np.log1p(12000000000)
    assert abs(result["log_gas_price"].iloc[0] - expected) < 1e-6


def test_load_and_clean_saves_csv(tmp_path):
    df = _minimal_df()
    zip_path = _make_zip(tmp_path, df)
    out_path = str(tmp_path / "out.csv")

    load_and_clean(zip_path, out_path, _default_config())

    assert (tmp_path / "out.csv").exists()


def test_load_and_clean_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_and_clean(str(tmp_path / "missing.zip"), str(tmp_path / "out.csv"), _default_config())
