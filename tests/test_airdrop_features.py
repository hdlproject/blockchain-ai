import math
from datetime import datetime, timezone
from unittest.mock import MagicMock
from blockchain_ai.feature.airdrop_features import compute_airdrop_features, derive_funder, AirdropFeatureExtractor

_ADDR = "0xwallet000000000000000000000000000000001"
_BASE_TS = int(datetime.now(timezone.utc).timestamp())

# wallet that received a token and immediately dumped it — sybil pattern
_TXS_SYBIL = [
    # funded by a shared funder
    {"from": "0xfunder", "to": _ADDR, "value": "1000000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 100), "gasPrice": "20000000000"},
    # unrelated activity after receiving the token
    {"from": _ADDR, "to": "0xexchange", "value": "0",
     "isError": "0", "timeStamp": str(_BASE_TS + 120), "gasPrice": "20000000000"},
]

_TOKEN_TXS_SYBIL = [
    # received token1 at _BASE_TS + 60
    {"contractAddress": "0xtoken1", "from": "0xdistributor", "to": _ADDR,
     "value": "1000", "timeStamp": str(_BASE_TS + 60)},
    # dumped token1 30 minutes later
    {"contractAddress": "0xtoken1", "from": _ADDR, "to": "0xexchange",
     "value": "1000", "timeStamp": str(_BASE_TS + 1860)},
]

# wallet with long history that held its token for months — genuine pattern
_TXS_GENUINE = [
    {"from": "0xother1", "to": _ADDR, "value": "500000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 86400 * 365), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother2", "value": "100000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 86400 * 300), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother3", "value": "100000000000000000",
     "isError": "0", "timeStamp": str(_BASE_TS - 86400 * 200), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother4", "value": "0",
     "isError": "0", "timeStamp": str(_BASE_TS + 86400 * 10), "gasPrice": "20000000000"},
]

_TOKEN_TXS_GENUINE = [
    # first-ever token receipt: token1 at -100 days (anchor for tx_count_before_first_inflow)
    {"contractAddress": "0xtoken1", "from": "0xother1", "to": _ADDR,
     "value": "1000", "timeStamp": str(_BASE_TS - 86400 * 100)},
    # second token received later
    {"contractAddress": "0xtoken2", "from": "0xother2", "to": _ADDR,
     "value": "2000", "timeStamp": str(_BASE_TS - 86400 * 50)},
    # token1 finally sent out 140 days after it was received
    {"contractAddress": "0xtoken1", "from": _ADDR, "to": "0xexchange",
     "value": "500", "timeStamp": str(_BASE_TS + 86400 * 40)},
]

_NO_FUNDER_LOOKUP = lambda funder: 0  # noqa: E731


def test_wallet_age_days_sybil_is_short():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    assert f["wallet_age_days"] < 1


def test_wallet_age_days_genuine_is_long():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    assert f["wallet_age_days"] > 300


def test_tx_count_before_first_inflow_sybil_is_one():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    # only the funding tx at _BASE_TS - 100 is before the token receipt at _BASE_TS + 60
    assert f["tx_count_before_first_inflow"] == 1


def test_tx_count_before_first_inflow_genuine_is_three():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    # three txs happen before the first token receipt at -100 days
    assert f["tx_count_before_first_inflow"] == 3


def test_tx_count_before_first_inflow_no_token_receipt_uses_total_tx_count():
    txs = [
        {"from": _ADDR, "to": "0xa", "value": "0", "isError": "0",
         "timeStamp": str(_BASE_TS), "gasPrice": "1"},
        {"from": _ADDR, "to": "0xb", "value": "0", "isError": "0",
         "timeStamp": str(_BASE_TS + 60), "gasPrice": "1"},
    ]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["tx_count_before_first_inflow"] == 2.0


def test_token_type_diversity():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    assert f["token_type_diversity"] == 1

    f2 = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    assert f2["token_type_diversity"] == 2


def test_inflow_to_outflow_hours_sybil_is_short():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    # received at +60, dumped at +1860 → 1800s = 0.5 hours
    assert abs(f["inflow_to_outflow_hours"] - 0.5) < 0.01


def test_inflow_to_outflow_hours_genuine_is_long():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _NO_FUNDER_LOOKUP)
    # token1 received at -100 days, sent at +40 days → 140 days = 3360 hours
    assert f["inflow_to_outflow_hours"] > 3000


def test_inflow_to_outflow_hours_no_token_activity_returns_zero():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
            "isError": "0", "timeStamp": str(_BASE_TS + 100), "gasPrice": "20000000000"}]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["inflow_to_outflow_hours"] == 0.0


def test_shared_funder_score_uses_log1p_of_lookup():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, lambda funder: 3)
    assert abs(f["shared_funder_score"] - math.log1p(3)) < 1e-9


def test_shared_funder_score_zero_when_lookup_returns_zero():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    assert f["shared_funder_score"] == 0.0


def test_shared_funder_score_zero_when_no_funder():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
            "isError": "0", "timeStamp": str(_BASE_TS + 100), "gasPrice": "20000000000"}]
    f = compute_airdrop_features(_ADDR, txs, [], lambda funder: 5)
    assert f["shared_funder_score"] == 0.0


def test_inter_tx_time_variance_single_tx_is_zero():
    txs = [_TXS_SYBIL[0]]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["inter_tx_time_variance"] == 0.0


def test_inter_tx_time_variance_regular_spacing_is_zero():
    base = _BASE_TS
    txs = [
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base), "gasPrice": "20000000000"},
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base + 60), "gasPrice": "20000000000"},
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base + 120), "gasPrice": "20000000000"},
    ]
    f = compute_airdrop_features(_ADDR, txs, [], _NO_FUNDER_LOOKUP)
    assert f["inter_tx_time_variance"] == 0.0


def test_unique_counterparty_count():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _NO_FUNDER_LOOKUP)
    # counterparties: 0xfunder, 0xexchange = 2
    assert f["unique_counterparty_count"] == 2.0


def test_derive_funder_returns_earliest_inbound_value_sender():
    assert derive_funder(_ADDR, _TXS_SYBIL) == "0xfunder"


def test_derive_funder_returns_none_when_no_inbound_value():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
            "isError": "0", "timeStamp": str(_BASE_TS), "gasPrice": "1"}]
    assert derive_funder(_ADDR, txs) is None


def test_extractor_calls_etherscan_records_funder_and_delegates():
    client = MagicMock()
    client.get_tx_list.return_value = _TXS_SYBIL
    client.get_token_transfers.return_value = _TOKEN_TXS_SYBIL
    ledger = MagicMock()
    ledger.funded_count.return_value = 0
    extractor = AirdropFeatureExtractor(client, ledger)
    features = extractor.extract(_ADDR)
    client.get_tx_list.assert_called_once_with(_ADDR)
    client.get_token_transfers.assert_called_once_with(_ADDR)
    ledger.record.assert_called_once_with("0xfunder", _ADDR)
    assert "wallet_age_days" in features


def test_empty_txs_returns_zero_features():
    f = compute_airdrop_features(_ADDR, [], [], _NO_FUNDER_LOOKUP)
    assert all(v == 0.0 for v in f.values())
