from unittest.mock import patch, MagicMock
from blockchain_ai.connector.goplus import GoPlusClient
from blockchain_ai.connector.schema import AddressRecord, TokenRecord
from blockchain_ai.config import GoPlusConfig


def _client():
    return GoPlusClient("https://api.gopluslabs.io/api/v1", chain_id=1, rate_limit_per_sec=100, timeout_sec=10)


def _mock_get(json_data):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = json_data
    return m


def test_from_config():
    cfg = GoPlusConfig(base_url="https://api.gopluslabs.io/api/v1", chain_id=1, rate_limit_per_sec=2, timeout_sec=30)
    c = GoPlusClient.from_config(cfg)
    assert c._chain_id == 1


def test_get_token_security_honeypot_sets_flag():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"0xtoken1": {"is_honeypot": "1", "is_open_source": "1", "is_blacklisted": "0",
                                "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
                                "is_mintable": "0", "transfer_pausable": "0"}},
    })):
        records = _client().get_token_security(["0xtoken1"])
    assert len(records) == 1
    assert "honeypot" in records[0].flags
    assert records[0].is_risky is True
    assert records[0].sources == ["goplus"]


def test_get_token_security_safe_returns_no_flags():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"0xsafe": {"is_honeypot": "0", "is_open_source": "1", "is_blacklisted": "0",
                              "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
                              "is_mintable": "0", "transfer_pausable": "0"}},
    })):
        records = _client().get_token_security(["0xsafe"])
    assert records[0].flags == []
    assert records[0].is_risky is False


def test_get_token_security_no_source_code_is_flag():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"0xclosed": {"is_honeypot": "0", "is_open_source": "0", "is_blacklisted": "0",
                                "can_take_back_ownership": "0", "hidden_owner": "0", "selfdestruct": "0",
                                "is_mintable": "0", "transfer_pausable": "0"}},
    })):
        records = _client().get_token_security(["0xclosed"])
    assert "no_source_code" in records[0].flags


def test_get_address_security_phishing():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"blacklist_doubt": "0", "cybercrime": "0", "money_laundering": "0",
                   "phishing_activities": "1", "stealing_attack": "0"},
    })):
        record = _client().get_address_security("0xphisher")
    assert isinstance(record, AddressRecord)
    assert record.label == "scammer"
    assert "phishing_activities" in record.flags


def test_get_address_security_clean_returns_unknown():
    with patch("requests.get", return_value=_mock_get({
        "code": 1, "message": "ok",
        "result": {"blacklist_doubt": "0", "cybercrime": "0", "money_laundering": "0",
                   "phishing_activities": "0", "stealing_attack": "0"},
    })):
        record = _client().get_address_security("0xclean")
    assert record.label == "unknown"
    assert record.flags == []
