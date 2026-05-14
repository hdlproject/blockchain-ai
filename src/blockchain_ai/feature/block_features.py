import pandas as pd

from blockchain_ai.connector.etherscan import EtherscanClient
from blockchain_ai.feature.feature_extractor import FeatureExtractor


class BlockFeatureExtractor(FeatureExtractor):
    TREND_LOOKBACK = 10

    def __init__(self, client: EtherscanClient | None = None):
        self._client = client

    def extract(self, rows: list[dict] | None = None) -> pd.DataFrame:
        if rows is None:
            if self._client is None:
                raise RuntimeError("No Etherscan client configured")

            latest = self._client.get_latest_block_number()
            rows = [self._client.get_block(n) for n in range(latest - self.TREND_LOOKBACK, latest + 1)]
            rows = [r for r in rows if r]
            if not rows:
                raise RuntimeError("Could not fetch recent blocks from Etherscan")

        df = pd.DataFrame(rows)
        df["base_fee_gwei"] = df["base_fee_per_gas"] / 1e9
        df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.dayofweek
        shifted = df["base_fee_gwei"].shift(self.TREND_LOOKBACK)
        df["base_fee_trend"] = ((df["base_fee_gwei"] - shifted) / shifted).fillna(0.0)
        return df
