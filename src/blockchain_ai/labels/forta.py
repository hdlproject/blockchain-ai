import requests
from datetime import datetime, timezone
from blockchain_ai.config import FortaConfig
from blockchain_ai.labels.schema import AddressRecord

_QUERY = """
query GetAlerts($input: AlertsInput) {
  alerts(input: $input) {
    alerts { hash name severity addresses createdAt }
    pageInfo { hasNextPage endCursor { alertId blockNumber } }
  }
}
"""


class FortaClient:
    def __init__(self, graphql_url: str, timeout_sec: int, max_alerts: int, scam_bot_ids: list[str]):
        self._url = graphql_url
        self._timeout = timeout_sec
        self._max_alerts = max_alerts
        self._bot_ids = scam_bot_ids

    @classmethod
    def from_config(cls, config: FortaConfig) -> "FortaClient":
        return cls(config.graphql_url, config.timeout_sec, config.max_alerts, config.scam_bot_ids)

    def _query(self, variables: dict) -> dict:
        response = requests.post(self._url, json={"query": _QUERY, "variables": variables},
                                 headers={"Content-Type": "application/json"}, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"Forta GraphQL error: {data['errors']}")
        return data["data"]["alerts"]

    def fetch_scam_addresses(self) -> list[AddressRecord]:
        seen: dict[str, AddressRecord] = {}
        now = datetime.now(timezone.utc).isoformat()
        end_cursor = None
        collected = 0
        while collected < self._max_alerts:
            variables: dict = {"input": {"bots": self._bot_ids, "first": min(100, self._max_alerts - collected)}}
            if end_cursor:
                variables["input"]["after"] = end_cursor
            result = self._query(variables)
            alerts = result["alerts"]
            collected += len(alerts)
            for alert in alerts:
                label = _severity_to_label(alert.get("severity", ""))
                confidence = _severity_to_confidence(alert.get("severity", ""))
                flag = alert.get("name", "unknown").lower().replace(" ", "_")
                for addr in alert.get("addresses") or []:
                    addr = addr.lower()
                    if addr.startswith("0x") and len(addr) == 42 and addr not in seen:
                        seen[addr] = AddressRecord(
                            address=addr, chain_id=1, label=label, confidence=confidence,
                            sources=["forta"], flags=[flag], fetched_at=now,
                        )
            if not result["pageInfo"]["hasNextPage"]:
                break
            end_cursor = result["pageInfo"]["endCursor"]
        return list(seen.values())


def _severity_to_label(severity: str) -> str:
    return {"CRITICAL": "scammer", "HIGH": "scammer", "MEDIUM": "phishing"}.get(severity, "suspicious")


def _severity_to_confidence(severity: str) -> float:
    return {"CRITICAL": 0.9, "HIGH": 0.8, "MEDIUM": 0.6}.get(severity, 0.4)
