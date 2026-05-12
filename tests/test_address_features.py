from unittest.mock import MagicMock
from blockchain_ai.feature.address_features import AddressFeatureExtractor, _compute_features, _entropy

_ADDRESS = "0xabc0000000000000000000000000000000000001"

_TXS = [
    {"from": _ADDRESS, "to": "0xother1", "value": "1000000000000000000",
     "isError": "0", "timeStamp": "1700000000", "gasPrice": "20000000000",
     "contractAddress": ""},
    {"from": "0xother2", "to": _ADDRESS, "value": "500000000000000000",
     "isError": "0", "timeStamp": "1700003600", "gasPrice": "25000000000",
     "contractAddress": ""},
    {"from": _ADDRESS, "to": "", "value": "0",
     "isError": "1", "timeStamp": "1700007200", "gasPrice": "15000000000",
     "contractAddress": _ADDRESS},
]

_TOKEN_TXS = [
    {"contractAddress": "0xtoken1", "from": _ADDRESS, "to": "0xother3", "value": "100", "timeStamp": "1700000000"},
    {"contractAddress": "0xtoken2", "from": "0xother4", "to": _ADDRESS, "value": "200", "timeStamp": "1700000001"},
]


def test_extract_calls_etherscan_methods():
    client = MagicMock()
    client.get_tx_list.return_value = _TXS
    client.get_token_transfers.return_value = _TOKEN_TXS
    extractor = AddressFeatureExtractor(client)
    features = extractor.extract(_ADDRESS)
    client.get_tx_list.assert_called_once_with(_ADDRESS)
    client.get_token_transfers.assert_called_once_with(_ADDRESS)
    assert isinstance(features, dict)


def test_tx_count():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["tx_count"] == 3.0


def test_failed_tx_ratio():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["failed_tx_ratio"] == pytest.approx(1/3, abs=0.001)


def test_contract_creation_count():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["contract_creation_count"] == 1.0


def test_erc20_token_count():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["erc20_token_count"] == 2.0


def test_is_contract_detected():
    features = _compute_features(_ADDRESS, _TXS, _TOKEN_TXS)
    assert features["is_contract"] == 1.0


def test_empty_txs_returns_zero_features():
    features = _compute_features(_ADDRESS, [], [])
    assert features["tx_count"] == 0.0
    assert features["account_age_days"] == 0.0


def test_entropy_uniform_distribution():
    values = list(range(8)) * 2  # 8 distinct hours, each appearing twice
    e = _entropy(values, 24)
    assert e == pytest.approx(3.0, abs=0.01)  # log2(8) = 3


def test_entropy_single_value():
    values = [5] * 10  # always hour 5
    e = _entropy(values, 24)
    assert e == pytest.approx(0.0, abs=0.001)


import pytest
