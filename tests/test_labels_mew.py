from unittest.mock import patch, MagicMock
from blockchain_ai.connector.mew import MEWClient
from blockchain_ai.config import MEWConfig

_SAMPLE_RESPONSE = [
    {"address": "0xabcdef1234567890abcdef1234567890abcdef12", "comment": "XRP phishing site", "date": "2024-01-01"},
    {"address": "0x1111111111111111111111111111111111111111", "comment": "Known scam", "date": "2024-01-02"},
    {"address": "notanaddress", "comment": "bad", "date": "2024-01-03"},
    {"address": "0xshort", "comment": "bad", "date": "2024-01-04"},
    {"comment": "missing address key", "date": "2024-01-05"},
]


def _client():
    return MEWClient(url="https://example.com/darklist.json", timeout_sec=10)


def _mock_response(data):
    m = MagicMock()
    m.json.return_value = data
    m.raise_for_status = MagicMock()
    return m


def test_from_config():
    cfg = MEWConfig(url="https://example.com/darklist.json", timeout_sec=30)
    c = MEWClient.from_config(cfg)
    assert c._url == "https://example.com/darklist.json"


def test_returns_valid_eth_addresses_only():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_RESPONSE)):
        records = _client().fetch_eth_addresses()
    assert len(records) == 2


def test_label_and_source():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_RESPONSE)):
        records = _client().fetch_eth_addresses()
    for r in records:
        assert r.label == "phishing"
        assert r.sources == ["mew"]
        assert r.flags == ["phishing"]
        assert r.confidence == 0.9


def test_address_normalized_to_lowercase():
    with patch("requests.get", return_value=_mock_response(
        [{"address": "0xABCDEF1234567890ABCDEF1234567890ABCDEF12", "comment": "", "date": ""}]
    )):
        records = _client().fetch_eth_addresses()
    assert records[0].address == "0xabcdef1234567890abcdef1234567890abcdef12"
