from unittest.mock import MagicMock

import pandas as pd
import pytest

from blockchain_ai.feature.block_features import BlockFeatureExtractor

_ROWS = [
    {"block_number": 100 + i, "base_fee_per_gas": (10 + i) * 10**9,
     "gas_used_ratio": 0.5, "timestamp": 1_700_000_000 + i * 12}
    for i in range(15)
]


def test_extract_adds_feature_columns():
    extractor = BlockFeatureExtractor()
    df = extractor.extract(_ROWS)
    assert "base_fee_gwei" in df.columns
    assert "hour_of_day" in df.columns
    assert "day_of_week" in df.columns
    assert "base_fee_trend" in df.columns


def test_extract_base_fee_gwei_conversion():
    extractor = BlockFeatureExtractor()
    df = extractor.extract(_ROWS)
    assert df["base_fee_gwei"].iloc[0] == pytest.approx(10.0, abs=1e-6)


def test_extract_trend_zero_within_lookback():
    extractor = BlockFeatureExtractor()
    df = extractor.extract(_ROWS)
    # First TREND_LOOKBACK rows have no prior reference → fillna(0.0)
    assert df["base_fee_trend"].iloc[0] == pytest.approx(0.0)


def test_extract_trend_nonzero_after_lookback():
    extractor = BlockFeatureExtractor()
    df = extractor.extract(_ROWS)
    assert df["base_fee_trend"].iloc[BlockFeatureExtractor.TREND_LOOKBACK] != 0.0


def test_extract_raises_without_client():
    extractor = BlockFeatureExtractor()
    with pytest.raises(RuntimeError, match="No Etherscan client"):
        extractor.extract()


def test_extract_calls_client_and_returns_dataframe():
    client = MagicMock()
    client.get_latest_block_number.return_value = 114
    client.get_block.side_effect = lambda n: {
        "block_number": n,
        "base_fee_per_gas": 10 * 10**9,
        "gas_used_ratio": 0.5,
        "timestamp": 1_700_000_000 + n * 12,
    }
    extractor = BlockFeatureExtractor(client)
    df = extractor.extract()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == BlockFeatureExtractor.TREND_LOOKBACK + 1
    assert "base_fee_gwei" in df.columns


def test_extract_raises_when_no_blocks_returned():
    client = MagicMock()
    client.get_latest_block_number.return_value = 200
    client.get_block.return_value = None
    extractor = BlockFeatureExtractor(client)
    with pytest.raises(RuntimeError, match="Could not fetch recent blocks"):
        extractor.extract()
