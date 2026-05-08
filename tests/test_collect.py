import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from collect_blocks import apply_target_shift
from blockchain_ai.block_features import BlockFeatureExtractor

_extractor = BlockFeatureExtractor()


def _raw_rows(n=15):
    return [
        {
            "block_number": 1000 + i,
            "base_fee_per_gas": (10 + i) * 10**9,  # 10–24 Gwei in Wei
            "gas_used_ratio": 0.5 + (i % 5) * 0.1,
            # unix timestamp: 2024-01-01 00:00:00 UTC + 12s per block
            "timestamp": 1704067200 + i * 12,
        }
        for i in range(n)
    ]


def test_derive_features_adds_base_fee_gwei(tmp_path):
    rows = _raw_rows()
    df = _extractor.derive(rows)
    assert "base_fee_gwei" in df.columns
    assert abs(df["base_fee_gwei"].iloc[0] - 10.0) < 1e-6


def test_derive_features_preserves_gas_used_ratio(tmp_path):
    rows = _raw_rows()
    df = _extractor.derive(rows)
    assert "gas_used_ratio" in df.columns
    assert df["gas_used_ratio"].iloc[0] == pytest.approx(0.5)


def test_derive_features_adds_hour_of_day(tmp_path):
    rows = _raw_rows()
    df = _extractor.derive(rows)
    assert "hour_of_day" in df.columns
    assert 0 <= df["hour_of_day"].iloc[0] <= 23


def test_derive_features_adds_day_of_week(tmp_path):
    rows = _raw_rows()
    df = _extractor.derive(rows)
    assert "day_of_week" in df.columns
    assert 0 <= df["day_of_week"].iloc[0] <= 6


def test_derive_features_adds_base_fee_trend(tmp_path):
    rows = _raw_rows(15)
    df = _extractor.derive(rows)
    assert "base_fee_trend" in df.columns
    # first 10 rows have no 10-block lookback — filled with 0.0
    assert df["base_fee_trend"].iloc[0] == pytest.approx(0.0)
    # row 10 has a valid trend
    assert df["base_fee_trend"].iloc[10] != pytest.approx(0.0)


def test_apply_target_shift_adds_target_column():
    df = pd.DataFrame({"base_fee_gwei": [10.0, 11.0, 12.0, 13.0]})
    result = apply_target_shift(df, source_col="base_fee_gwei", target_col="next_base_fee_gwei")
    assert "next_base_fee_gwei" in result.columns
    # next_base_fee_gwei[0] = base_fee_gwei[1] = 11.0
    assert result["next_base_fee_gwei"].iloc[0] == pytest.approx(11.0)
    # source column must not be overwritten
    assert result["base_fee_gwei"].iloc[0] == pytest.approx(10.0)


def test_apply_target_shift_drops_last_row():
    df = pd.DataFrame({"base_fee_gwei": [10.0, 11.0, 12.0, 13.0]})
    result = apply_target_shift(df, source_col="base_fee_gwei", target_col="next_base_fee_gwei")
    assert len(result) == 3


def test_apply_target_shift_raises_if_too_few_rows():
    df = pd.DataFrame({"base_fee_gwei": [10.0]})
    with pytest.raises(RuntimeError, match="too few"):
        apply_target_shift(df, source_col="base_fee_gwei", target_col="base_fee_gwei")
