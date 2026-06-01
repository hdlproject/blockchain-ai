import math
import pandas as pd
import pytest
from blockchain_ai.feature.transaction_features import TransactionFeatureExtractor, KNOWN_TOKEN_DECIMALS


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
        "gas_price_gwei", "gas_used", "input_data_len",
        "hour_of_day", "sender_tx_count_window", "receiver_tx_count_window",
        "tx_type", "contract_type", "log_transfer_value",
    }
    assert expected.issubset(set(df.columns))


def test_log_transfer_value_zero_for_no_calldata_no_eth():
    txs = [_make_tx(value_wei="0", input_data="0x")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["log_transfer_value"].iloc[0] == 0.0


def test_contract_type_eth_transfer():
    txs = [_make_tx(input_data="0x")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["contract_type"].iloc[0] == 0.0


def test_contract_type_known_token():
    usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    txs = [_make_tx(to=usdc, input_data="0xabcdef")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["contract_type"].iloc[0] == 1.0


def test_contract_type_unknown_contract():
    unknown = "0x1234567890abcdef1234567890abcdef12345678"
    txs = [_make_tx(to=unknown, input_data="0xabcdef")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["contract_type"].iloc[0] == 2.0


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


def test_tx_type_eth_transfer():
    txs = [_make_tx(input_data="0x")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["tx_type"].iloc[0] == 0.0


def test_tx_type_token_transfer():
    # transfer(address,uint256) selector + 64 bytes to + 64 bytes amount
    selector = "a9059cbb"
    calldata = "0x" + selector + "0" * 128
    txs = [_make_tx(input_data=calldata)]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["tx_type"].iloc[0] == 1.0


def test_tx_type_approve():
    selector = "095ea7b3"
    calldata = "0x" + selector + "0" * 128
    txs = [_make_tx(input_data=calldata)]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["tx_type"].iloc[0] == 2.0


def test_tx_type_other():
    txs = [_make_tx(input_data="0xdeadbeef")]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["tx_type"].iloc[0] == 3.0




def test_log_transfer_value_eth():
    txs = [_make_tx(value_wei="1000000000000000000", input_data="0x")]  # 1 ETH
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert abs(df["log_transfer_value"].iloc[0] - math.log1p(1.0)) < 1e-5


def test_log_transfer_value_known_token():
    usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"  # 6 decimals
    # transfer(address,uint256): 100 USDC = 100 * 10^6 = 100_000_000
    amount = 100_000_000
    selector = "a9059cbb"
    calldata = "0x" + selector + "0" * 64 + hex(amount)[2:].zfill(64)
    txs = [_make_tx(to=usdc, value_wei="0", input_data=calldata)]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    expected = math.log1p(100.0)  # 100 USDC normalized
    assert abs(df["log_transfer_value"].iloc[0] - expected) < 1e-4


def test_log_transfer_value_unknown_token_is_zero():
    unknown = "0x1234567890abcdef1234567890abcdef12345678"
    selector = "a9059cbb"
    calldata = "0x" + selector + "0" * 128
    txs = [_make_tx(to=unknown, value_wei="0", input_data=calldata)]
    df = TransactionFeatureExtractor().extract_dataset(txs)
    assert df["log_transfer_value"].iloc[0] == 0.0
