from unittest.mock import patch, MagicMock
from blockchain_ai.connector.scamsniffer import ScamSnifferClient
from blockchain_ai.config import ScamSnifferConfig

_SAMPLE_RESPONSE = [
    "0xabcdef1234567890abcdef1234567890abcdef12",
    "0x1111111111111111111111111111111111111111",
    "notanaddress",
    "0xshort",
]


def _client():
    return ScamSnifferClient(url="https://example.com/address.json", timeout_sec=10)


def _mock_response(data):
    m = MagicMock()
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def test_from_config():
    cfg = ScamSnifferConfig(url="https://example.com/address.json", timeout_sec=30)
    c = ScamSnifferClient.from_config(cfg)
    assert c._url == "https://example.com/address.json"


def test_returns_valid_eth_addresses_only():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_RESPONSE)):
        records = _client().fetch_eth_addresses()
    assert len(records) == 2


def test_label_and_source():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_RESPONSE)):
        records = _client().fetch_eth_addresses()
    for r in records:
        assert r.label == "phishing"
        assert r.sources == ["scamsniffer"]
        assert r.flags == ["phishing"]
        assert r.confidence == 0.9


def test_address_normalized_to_lowercase():
    with patch("requests.get", return_value=_mock_response(["0xABCDEF1234567890ABCDEF1234567890ABCDEF12"])):
        records = _client().fetch_eth_addresses()
    assert records[0].address == "0xabcdef1234567890abcdef1234567890abcdef12"
