import os
import time
import requests
from blockchain_ai.config import EtherscanConfig


class EtherscanClient:
    def __init__(self, base_url: str, chain_id: int, rate_limit_per_sec: int, timeout_sec: int):
        api_key = os.environ.get("ETHERSCAN_API_KEY")
        if not api_key:
            raise RuntimeError("ETHERSCAN_API_KEY environment variable is not set")
        self._api_key = api_key
        self._base_url = base_url
        self._chain_id = chain_id
        self._sleep_secs = 1.0 / rate_limit_per_sec
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: EtherscanConfig) -> "EtherscanClient":
        return cls(
            base_url=config.base_url,
            chain_id=config.chain_id,
            rate_limit_per_sec=config.rate_limit_per_sec,
            timeout_sec=config.timeout_sec,
        )

    def _get(self, params: dict, _retries: int = 3) -> dict:
        params["apikey"] = self._api_key
        params["chainid"] = self._chain_id
        for attempt in range(_retries):
            time.sleep(self._sleep_secs)
            try:
                response = requests.get(self._base_url, params=params, timeout=self._timeout)
            except requests.exceptions.Timeout:
                if attempt < _retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            if response.status_code != 200:
                raise RuntimeError(f"Etherscan HTTP error: {response.status_code}")
            data = response.json()
            if str(data.get("status")) == "0":
                raise RuntimeError(f"Etherscan API error: {data.get('message')}")
            return data["result"]

    def get_latest_block_number(self) -> int:
        result = self._get({"module": "proxy", "action": "eth_blockNumber"})
        return int(result, 16)

    def get_block(self, block_number: int) -> dict | None:
        result = self._get({
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(block_number),
            "boolean": "false",
        })
        if result is None:
            return None
        base_fee_hex = result.get("baseFeePerGas")
        if not base_fee_hex:
            return None  # pre-EIP-1559 block
        base_fee = int(base_fee_hex, 16)
        if base_fee == 0:
            return None  # pre-EIP-1559 block
        gas_used = int(result["gasUsed"], 16)
        gas_limit = int(result["gasLimit"], 16)
        return {
            "block_number": block_number,
            "base_fee_per_gas": base_fee,
            "gas_used_ratio": gas_used / gas_limit if gas_limit > 0 else 0.0,
            "timestamp": int(result["timestamp"], 16),
        }
