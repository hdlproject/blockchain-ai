import pandas as pd
import pytest
from blockchain_ai.feature.transaction_features import TransactionFeatureExtractor


def _make_tx(frm="0xaaa", to="0xbbb", value_wei="1000000000000000000",
             gas="21000", gas_price="20000000000", input_data="0x",
             timestamp="1700000000"):
    return {
        "from": frm, "to": to, "value": value_wei,
        "gas": gas, "gasPrice": gas_price,
        "input": input_data, "timeStamp": timestamp,
    }


def test_extract_dataset_returns_dataframe():
    txs = [_make_tx() for _ in range(5)]
    extractor = TransactionFeatureExtractor()
    df = extractor.extract_dataset(txs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_extract_dataset_has_expected_columns():
    txs = [_make_tx()]
    extractor = TransactionFeatureExtractor()
    df = extractor.extract_dataset(txs)
    expected = {
        "value_eth", "gas_price_gwei", "gas_used", "input_data_len",
        "is_contract_call", "hour_of_day", "sender_tx_count_window",
        "sender_avg_value_eth", "receiver_tx_count_window",
    }
    assert expected.issubset(set(df.columns))


def test_value_eth_conversion():
    txs = [_make_tx(value_wei="2000000000000000000")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert abs(df["value_eth"].iloc[0] - 2.0) < 1e-6


def test_is_contract_call_with_calldata():
    txs = [_make_tx(input_data="0xabcdef")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["is_contract_call"].iloc[0] == 1.0


def test_is_contract_call_simple_transfer():
    txs = [_make_tx(input_data="0x")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["is_contract_call"].iloc[0] == 0.0


def test_sender_tx_count_window():
    txs = [_make_tx(frm="0xaaa") for _ in range(3)] + [_make_tx(frm="0xbbb")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    aaa_rows = df[df["sender_tx_count_window"] == 3.0]
    assert len(aaa_rows) == 3


def test_skips_tx_without_from():
    txs = [{"to": "0xbbb", "value": "0", "gas": "21000",
             "gasPrice": "1000000000", "input": "0x", "timeStamp": "1700000000"}]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert len(df) == 0


def test_input_data_len():
    # "0xabcd" = 2 bytes
    txs = [_make_tx(input_data="0xabcd")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["input_data_len"].iloc[0] == 2.0
