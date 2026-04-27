import os
import time
from unittest.mock import patch, MagicMock
import pytest
from blockchain_ai.etherscan import EtherscanClient


@pytest.fixture
def client():
    with patch.dict(os.environ, {"ETHERSCAN_API_KEY": "testkey"}):
        return EtherscanClient(
            base_url="https://api.etherscan.io/api",
            rate_limit_per_sec=100,  # high limit so tests don't sleep
            timeout_sec=10,
        )


def _ok_response(result):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "1", "message": "OK", "result": result}
    return m


def _error_response(message):
    m = MagicMock()
    m.status_code = 200
    m.json.return_value = {"status": "0", "message": message, "result": None}
    return m


def test_client_reads_api_key_from_env():
    with patch.dict(os.environ, {"ETHERSCAN_API_KEY": "mykey123"}):
        c = EtherscanClient("https://api.etherscan.io/api", 5, 10)
        assert c._api_key == "mykey123"


def test_client_raises_if_api_key_missing():
    env = {k: v for k, v in os.environ.items() if k != "ETHERSCAN_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="ETHERSCAN_API_KEY"):
            EtherscanClient("https://api.etherscan.io/api", 5, 10)


def test_get_latest_block_number(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response("0xf4240")
        result = client.get_latest_block_number()
        assert result == 0xf4240
        assert mock_get.called


def test_get_latest_block_number_raises_on_http_error(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        m = MagicMock()
        m.status_code = 429
        mock_get.return_value = m
        with pytest.raises(RuntimeError, match="429"):
            client.get_latest_block_number()


def test_get_latest_block_number_raises_on_api_error(client):
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("Invalid API Key")
        with pytest.raises(RuntimeError, match="Invalid API Key"):
            client.get_latest_block_number()


def test_get_fee_history_returns_list_of_dicts(client):
    fee_history_result = {
        "oldestBlock": "0xf4230",
        "baseFeePerGas": ["0x6fc23ac00", "0x6d54e9800", "0x0"],
        "gasUsedRatio": [0.95, 0.50],
    }
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(fee_history_result)
        result = client.get_fee_history(block_count=2, newest_block=0xf4240)
        # last entry in baseFeePerGas is the next block prediction — excluded
        assert len(result) == 2
        assert "base_fee_per_gas" in result[0]
        assert "gas_used_ratio" in result[0]
        assert "block_number" in result[0]


def test_get_fee_history_skips_pre_eip1559_blocks(client):
    fee_history_result = {
        "oldestBlock": "0xf4230",
        "baseFeePerGas": ["0x0", "0x6d54e9800", "0x0"],
        "gasUsedRatio": [0.95, 0.50],
    }
    with patch("blockchain_ai.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(fee_history_result)
        result = client.get_fee_history(block_count=2, newest_block=0xf4240)
        # block with baseFeePerGas == 0x0 (first entry) is pre-EIP-1559 — skipped
        assert all(r["base_fee_per_gas"] > 0 for r in result)


def test_rate_limiting_sleeps_between_calls(client):
    slow_client = EtherscanClient.__new__(EtherscanClient)
    slow_client._api_key = "testkey"
    slow_client._base_url = "https://api.etherscan.io/api"
    slow_client._sleep_secs = 0.2
    slow_client._timeout = 10

    with patch("blockchain_ai.etherscan.requests.get") as mock_get, \
         patch("blockchain_ai.etherscan.time.sleep") as mock_sleep:
        mock_get.return_value = _ok_response("0xf4240")
        slow_client.get_latest_block_number()
        mock_sleep.assert_called_once_with(0.2)
