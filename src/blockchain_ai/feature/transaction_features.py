from datetime import datetime, timezone
import pandas as pd


class TransactionFeatureExtractor:
    def extract_dataset(self, raw_txs: list[dict]) -> pd.DataFrame:
        sender_counts: dict[str, int] = {}
        sender_values: dict[str, list[float]] = {}
        receiver_counts: dict[str, int] = {}

        for tx in raw_txs:
            frm = tx.get("from", "").lower()
            to = tx.get("to", "").lower()
            value = int(tx.get("value", 0)) / 1e18
            if frm:
                sender_counts[frm] = sender_counts.get(frm, 0) + 1
                sender_values.setdefault(frm, []).append(value)
            if to:
                receiver_counts[to] = receiver_counts.get(to, 0) + 1

        sender_avg: dict[str, float] = {
            addr: sum(vals) / len(vals)
            for addr, vals in sender_values.items()
        }

        rows = [
            row for tx in raw_txs
            if (row := self._extract_one(tx, sender_counts, sender_avg, receiver_counts)) is not None
        ]
        return pd.DataFrame(rows)

    def _extract_one(
        self,
        tx: dict,
        sender_counts: dict[str, int],
        sender_avg: dict[str, float],
        receiver_counts: dict[str, int],
    ) -> dict | None:
        frm = tx.get("from", "").lower()
        to = tx.get("to", "").lower()
        if not frm:
            return None

        value_eth = int(tx.get("value", 0)) / 1e18
        gas_price_gwei = int(tx.get("gasPrice", 0)) / 1e9
        gas_used = float(int(tx.get("gas", 0)))
        input_data = tx.get("input", "0x") or "0x"
        input_data_len = float(max(0, (len(input_data) - 2) // 2)) if input_data != "0x" else 0.0
        is_contract_call = 1.0 if input_data != "0x" else 0.0
        timestamp = int(tx.get("timeStamp", tx.get("timestamp", 0)))
        hour_of_day = float(datetime.fromtimestamp(timestamp, tz=timezone.utc).hour) if timestamp else 0.0

        return {
            "value_eth": round(value_eth, 6),
            "gas_price_gwei": round(gas_price_gwei, 4),
            "gas_used": gas_used,
            "input_data_len": input_data_len,
            "is_contract_call": is_contract_call,
            "hour_of_day": hour_of_day,
            "sender_tx_count_window": float(sender_counts.get(frm, 1)),
            "sender_avg_value_eth": round(sender_avg.get(frm, value_eth), 6),
            "receiver_tx_count_window": float(receiver_counts.get(to, 1)),
        }
