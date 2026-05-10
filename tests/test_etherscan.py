import os
import time
from unittest.mock import patch, MagicMock
import pytest
from blockchain_ai.connector.etherscan import EtherscanClient


@pytest.fixture
def client():
    with patch.dict(os.environ, {"ETHERSCAN_API_KEY": "testkey"}):
        return EtherscanClient(
            base_url="https://api.etherscan.io/v2/api",
            chain_id=1,
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
        c = EtherscanClient("https://api.etherscan.io/v2/api", chain_id=1, rate_limit_per_sec=5, timeout_sec=10)
        assert c._api_key == "mykey123"


def test_client_raises_if_api_key_missing():
    env = {k: v for k, v in os.environ.items() if k != "ETHERSCAN_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(RuntimeError, match="ETHERSCAN_API_KEY"):
            EtherscanClient("https://api.etherscan.io/v2/api", chain_id=1, rate_limit_per_sec=5, timeout_sec=10)


def test_get_latest_block_number(client):
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response("0xf4240")
        result = client.get_latest_block_number()
        assert result == 0xf4240
        assert mock_get.called


def test_get_latest_block_number_raises_on_http_error(client):
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        m = MagicMock()
        m.status_code = 429
        mock_get.return_value = m
        with pytest.raises(RuntimeError, match="429"):
            client.get_latest_block_number()


def test_get_latest_block_number_raises_on_api_error(client):
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("Invalid API Key")
        with pytest.raises(RuntimeError, match="Invalid API Key"):
            client.get_latest_block_number()


def test_get_block_returns_dict(client):
    block_result = {
        "number": "0xf4240",
        "baseFeePerGas": "0x6fc23ac00",
        "gasUsed": "0x1e8480",
        "gasLimit": "0x3938700",
        "timestamp": "0x65a1b2c3",
    }
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(block_result)
        result = client.get_block(0xf4240)
        assert result is not None
        assert "base_fee_per_gas" in result
        assert "gas_used_ratio" in result
        assert "block_number" in result
        assert "timestamp" in result


def test_get_block_skips_pre_eip1559_blocks(client):
    block_result = {
        "number": "0xf4240",
        "baseFeePerGas": "0x0",
        "gasUsed": "0x1e8480",
        "gasLimit": "0x3938700",
        "timestamp": "0x65a1b2c3",
    }
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(block_result)
        result = client.get_block(0xf4240)
        assert result is None


def test_get_block_returns_none_for_missing_block(client):
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(None)
        result = client.get_block(0xf4240)
        assert result is None


def test_rate_limiting_sleeps_between_calls(client):
    slow_client = EtherscanClient.__new__(EtherscanClient)
    slow_client._api_key = "testkey"
    slow_client._base_url = "https://api.etherscan.io/v2/api"
    slow_client._chain_id = 1
    slow_client._sleep_secs = 0.2
    slow_client._timeout = 10

    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get, \
         patch("blockchain_ai.connector.etherscan.time.sleep") as mock_sleep:
        mock_get.return_value = _ok_response("0xf4240")
        slow_client.get_latest_block_number()
        mock_sleep.assert_called_once_with(0.2)


def test_get_tx_list_returns_list(client):
    txs = [{"hash": "0xabc", "from": "0x1", "to": "0x2", "value": "1000000000000000000",
             "isError": "0", "timeStamp": "1700000000", "gasPrice": "20000000000"}]
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(txs)
        result = client.get_tx_list("0xsome_address")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["hash"] == "0xabc"


def test_get_tx_list_returns_empty_on_no_transactions(client):
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("No transactions found")
        result = client.get_tx_list("0xempty")
    assert result == []


def test_get_token_transfers_returns_list(client):
    transfers = [{"contractAddress": "0xtoken", "from": "0x1", "to": "0x2",
                  "value": "1000", "timeStamp": "1700000000"}]
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _ok_response(transfers)
        result = client.get_token_transfers("0xsome_address")
    assert isinstance(result, list)
    assert result[0]["contractAddress"] == "0xtoken"


def test_get_token_transfers_returns_empty_on_no_transfers(client):
    with patch("blockchain_ai.connector.etherscan.requests.get") as mock_get:
        mock_get.return_value = _error_response("No transactions found")
        result = client.get_token_transfers("0xempty")
    assert result == []
