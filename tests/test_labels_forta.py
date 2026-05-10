from unittest.mock import patch, MagicMock
from blockchain_ai.connector.forta import FortaClient, _severity_to_label, _severity_to_confidence
from blockchain_ai.config import FortaConfig


def _client():
    return FortaClient("https://api.forta.network/graphql", timeout_sec=10, max_alerts=100, scam_bot_ids=["0xbot1"])


def _mock_post(alerts, has_next=False):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"data": {"alerts": {
        "alerts": alerts,
        "pageInfo": {"hasNextPage": has_next, "endCursor": {"alertId": "x", "blockNumber": 1}},
    }}}
    return m


def test_from_config():
    cfg = FortaConfig(graphql_url="https://api.forta.network/graphql", timeout_sec=30, max_alerts=500, scam_bot_ids=["0xabc"])
    c = FortaClient.from_config(cfg)
    assert c._max_alerts == 500


def test_extracts_addresses_from_alerts():
    alerts = [{"hash": "0xh1", "name": "Phishing", "severity": "HIGH",
               "addresses": ["0x1111111111111111111111111111111111111111"], "createdAt": "2026-01-01"}]
    with patch("requests.post", return_value=_mock_post(alerts)):
        records = _client().fetch_scam_addresses()
    assert len(records) == 1
    assert records[0].address == "0x1111111111111111111111111111111111111111"
    assert records[0].sources == ["forta"]


def test_deduplicates_same_address_across_alerts():
    addr = "0x2222222222222222222222222222222222222222"
    alerts = [
        {"hash": "0xh1", "name": "Scam A", "severity": "HIGH", "addresses": [addr], "createdAt": "2026-01-01"},
        {"hash": "0xh2", "name": "Scam B", "severity": "CRITICAL", "addresses": [addr], "createdAt": "2026-01-01"},
    ]
    with patch("requests.post", return_value=_mock_post(alerts)):
        records = _client().fetch_scam_addresses()
    assert len(records) == 1


def test_skips_non_eth_addresses():
    alerts = [{"hash": "0xh1", "name": "Scam", "severity": "HIGH",
               "addresses": ["not_an_address", "0x3333333333333333333333333333333333333333"], "createdAt": "2026-01-01"}]
    with patch("requests.post", return_value=_mock_post(alerts)):
        records = _client().fetch_scam_addresses()
    assert len(records) == 1


def test_severity_mapping():
    assert _severity_to_label("CRITICAL") == "scammer"
    assert _severity_to_label("HIGH") == "scammer"
    assert _severity_to_label("MEDIUM") == "phishing"
    assert _severity_to_label("LOW") == "suspicious"
    assert _severity_to_confidence("CRITICAL") == 0.9
    assert _severity_to_confidence("HIGH") == 0.8
    assert _severity_to_confidence("MEDIUM") == 0.6
