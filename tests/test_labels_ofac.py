from unittest.mock import patch, MagicMock
from blockchain_ai.labels.ofac import OFACFetcher
from blockchain_ai.config import OFACConfig

_SAMPLE_CSV = (
    "ent_num,alt_num,alt_type,alt_name,alt_remarks\n"
    "12345,1,Digital Currency Address - ETH,0xABCDEF1234567890ABCDEF1234567890ABCDEF12,\n"
    "12345,2,aka,Some Name,\n"
    "99999,1,Digital Currency Address - BTC,1BitcoinAddress,\n"
    "88888,1,Digital Currency Address - ETH,0x1111111111111111111111111111111111111111,\n"
)


def _fetcher():
    return OFACFetcher(alt_url="https://example.com/alt.csv", timeout_sec=10)


def _mock_response(text):
    m = MagicMock()
    m.text = text
    m.raise_for_status = MagicMock()
    return m


def test_from_config():
    cfg = OFACConfig(alt_url="https://example.com/alt.csv", timeout_sec=60)
    f = OFACFetcher.from_config(cfg)
    assert f._alt_url == "https://example.com/alt.csv"


def test_returns_only_eth_addresses():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    assert len(records) == 2


def test_excludes_btc_addresses():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    for r in records:
        assert r.address.startswith("0x")


def test_label_is_sanctioned_confidence_is_one():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    for r in records:
        assert r.label == "sanctioned"
        assert r.confidence == 1.0
        assert "ofac_sdn" in r.flags
        assert r.sources == ["ofac"]


def test_address_normalized_to_lowercase():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    for r in records:
        assert r.address == r.address.lower()
