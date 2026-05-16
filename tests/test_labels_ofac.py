from unittest.mock import patch, MagicMock
from blockchain_ai.connector.ofac import OFACFetcher
from blockchain_ai.config import OFACConfig

# Minimal sdn.csv rows: 12 columns, ETH addresses embedded in Remarks (col 11)
_SAMPLE_CSV = (
    "ent_num,SDN_Name,SDN_Type,Program,Title,Call_Sign,Vess_type,Tonnage,GRT,Vess_flag,Vess_owner,Remarks\n"
    "12345,Evil Corp,Individual,SDGT,-0-,-0-,-0-,-0-,-0-,-0-,-0-,"
    "\"Digital Currency Address - ETH 0xabcdef1234567890abcdef1234567890abcdef12; "
    "Digital Currency Address - ETH 0x1111111111111111111111111111111111111111;\"\n"
    "99999,No Crypto,Individual,SDGT,-0-,-0-,-0-,-0-,-0-,-0-,-0-,Some other remarks\n"
    "88888,Short Row,Individual\n"
)


def _fetcher():
    return OFACFetcher(sdn_url="https://example.com/sdn.csv", timeout_sec=10)


def _mock_response(text):
    m = MagicMock()
    m.text = text
    m.raise_for_status = MagicMock()
    return m


def test_from_config():
    cfg = OFACConfig(sdn_url="https://example.com/sdn.csv", timeout_sec=60)
    f = OFACFetcher.from_config(cfg)
    assert f._sdn_url == "https://example.com/sdn.csv"


def test_returns_only_eth_addresses():
    with patch("requests.get", return_value=_mock_response(_SAMPLE_CSV)):
        records = _fetcher().fetch_eth_addresses()
    assert len(records) == 2


def test_skips_rows_without_eth():
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
