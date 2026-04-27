import os
import time
import requests
from blockchain_ai.config import EtherscanConfig


class EtherscanClient:
    def __init__(self, base_url: str, rate_limit_per_sec: int, timeout_sec: int):
        api_key = os.environ.get("ETHERSCAN_API_KEY")
        if not api_key:
            raise RuntimeError("ETHERSCAN_API_KEY environment variable is not set")
        self._api_key = api_key
        self._base_url = base_url
        self._sleep_secs = 1.0 / rate_limit_per_sec
        self._timeout = timeout_sec

    @classmethod
    def from_config(cls, config: EtherscanConfig) -> "EtherscanClient":
        return cls(
            base_url=config.base_url,
            rate_limit_per_sec=config.rate_limit_per_sec,
            timeout_sec=config.timeout_sec,
        )

    def _get(self, params: dict) -> dict:
        time.sleep(self._sleep_secs)
        params["apikey"] = self._api_key
        response = requests.get(self._base_url, params=params, timeout=self._timeout)
        if response.status_code != 200:
            raise RuntimeError(f"Etherscan HTTP error: {response.status_code}")
        data = response.json()
        if str(data.get("status")) == "0":
            raise RuntimeError(f"Etherscan API error: {data.get('message')}")
        return data["result"]

    def get_latest_block_number(self) -> int:
        result = self._get({"module": "proxy", "action": "eth_blockNumber"})
        return int(result, 16)

    def get_fee_history(self, block_count: int, newest_block: int) -> list[dict]:
        result = self._get({
            "module": "proxy",
            "action": "eth_feeHistory",
            "blockCount": hex(block_count),
            "newestBlock": hex(newest_block),
            "rewardPercentiles": "",
        })
        oldest = int(result["oldestBlock"], 16)
        base_fees = result["baseFeePerGas"]
        ratios = result["gasUsedRatio"]

        # baseFeePerGas has block_count+1 entries (last is next-block prediction) — zip with ratios
        rows = []
        for i, (fee_hex, ratio) in enumerate(zip(base_fees, ratios)):
            base_fee = int(fee_hex, 16)
            if base_fee == 0:
                continue  # pre-EIP-1559 block
            rows.append({
                "block_number": oldest + i,
                "base_fee_per_gas": base_fee,
                "gas_used_ratio": float(ratio),
            })
        return rows
