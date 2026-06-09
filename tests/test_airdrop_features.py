from datetime import datetime, timezone
from unittest.mock import MagicMock
from blockchain_ai.feature.airdrop_features import compute_airdrop_features, AirdropFeatureExtractor

_ADDR = "0xwallet000000000000000000000000000000001"
_CONTRACT = "0xcontract00000000000000000000000000000001"
_AIRDROP_TS = 1_700_000_000  # 2023-11-14
_AIRDROP_DATE = datetime.fromtimestamp(_AIRDROP_TS, tz=timezone.utc)

# wallet that claimed and immediately dumped — sybil pattern
_TXS_SYBIL = [
    # funded by a shared funder before airdrop
    {"from": "0xfunder", "to": _ADDR, "value": "1000000000000000000",
     "isError": "0", "timeStamp": str(_AIRDROP_TS - 100), "gasPrice": "20000000000"},
    # claimed airdrop
    {"from": _ADDR, "to": _CONTRACT, "value": "0",
     "isError": "0", "timeStamp": str(_AIRDROP_TS + 60), "gasPrice": "20000000000"},
    # one more tx after claim
    {"from": _ADDR, "to": "0xexchange", "value": "0",
     "isError": "0", "timeStamp": str(_AIRDROP_TS + 120), "gasPrice": "20000000000"},
]

_TOKEN_TXS_SYBIL = [
    # outbound token transfer 30 minutes after claim
    {"contractAddress": "0xtoken1", "from": _ADDR, "to": "0xexchange",
     "value": "1000", "timeStamp": str(_AIRDROP_TS + 1860)},
]

# wallet with long history — genuine user pattern
_TXS_GENUINE = [
    {"from": "0xother1", "to": _ADDR, "value": "500000000000000000",
     "isError": "0", "timeStamp": str(_AIRDROP_TS - 86400 * 365), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother2", "value": "100000000000000000",
     "isError": "0", "timeStamp": str(_AIRDROP_TS - 86400 * 300), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": "0xother3", "value": "100000000000000000",
     "isError": "0", "timeStamp": str(_AIRDROP_TS - 86400 * 200), "gasPrice": "20000000000"},
    {"from": _ADDR, "to": _CONTRACT, "value": "0",
     "isError": "0", "timeStamp": str(_AIRDROP_TS + 86400 * 10), "gasPrice": "20000000000"},
]

_TOKEN_TXS_GENUINE = [
    {"contractAddress": "0xtoken1", "from": "0xother1", "to": _ADDR,
     "value": "1000", "timeStamp": str(_AIRDROP_TS - 86400 * 200)},
    {"contractAddress": "0xtoken2", "from": "0xother2", "to": _ADDR,
     "value": "2000", "timeStamp": str(_AIRDROP_TS - 86400 * 100)},
    # outbound token transfer 30 days after claim
    {"contractAddress": "0xtoken1", "from": _ADDR, "to": "0xexchange",
     "value": "500", "timeStamp": str(_AIRDROP_TS + 86400 * 40)},
]


def test_wallet_age_days_sybil_is_much_shorter_than_genuine():
    f_sybil = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, set())
    f_genuine = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _CONTRACT, _AIRDROP_DATE, set())
    # sybil wallet created 100s before airdrop; genuine wallet created 365 days before airdrop
    assert f_genuine["wallet_age_days"] - f_sybil["wallet_age_days"] > 300


def test_wallet_age_days_genuine_is_long():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _CONTRACT, _AIRDROP_DATE, set())
    assert f["wallet_age_days"] > 300


def test_tx_count_pre_airdrop_sybil_is_one():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, set())
    # only the funding tx at _AIRDROP_TS - 100 is before the airdrop timestamp
    assert f["tx_count_pre_airdrop"] == 1


def test_tx_count_pre_airdrop_genuine_is_positive():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _CONTRACT, _AIRDROP_DATE, set())
    assert f["tx_count_pre_airdrop"] == 3  # three txs before airdrop


def test_token_type_diversity():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, set())
    assert f["token_type_diversity"] == 1

    f2 = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _CONTRACT, _AIRDROP_DATE, set())
    assert f2["token_type_diversity"] == 2


def test_claim_to_withdraw_hours_sybil_is_short():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, set())
    # claim at _AIRDROP_TS + 60, first outbound token at _AIRDROP_TS + 1860 → 1800s = 0.5 hours
    assert abs(f["claim_to_withdraw_hours"] - 0.5) < 0.01


def test_claim_to_withdraw_hours_genuine_is_long():
    f = compute_airdrop_features(_ADDR, _TXS_GENUINE, _TOKEN_TXS_GENUINE, _CONTRACT, _AIRDROP_DATE, set())
    # claim at +10 days, first outbound token at +40 days → 30 days = 720 hours
    assert f["claim_to_withdraw_hours"] > 700


def test_claim_to_withdraw_hours_no_claim_returns_zero():
    txs = [{"from": _ADDR, "to": "0xother", "value": "0",
             "isError": "0", "timeStamp": str(_AIRDROP_TS + 100), "gasPrice": "20000000000"}]
    f = compute_airdrop_features(_ADDR, txs, [], _CONTRACT, _AIRDROP_DATE, set())
    assert f["claim_to_withdraw_hours"] == 0.0


def test_gas_source_shared_flagged():
    funding_set = {"0xfunder"}
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, funding_set)
    assert f["gas_source_shared"] == 1.0


def test_gas_source_shared_not_flagged():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, set())
    assert f["gas_source_shared"] == 0.0


def test_inter_tx_time_variance_single_tx_is_zero():
    txs = [_TXS_SYBIL[0]]
    f = compute_airdrop_features(_ADDR, txs, [], _CONTRACT, _AIRDROP_DATE, set())
    assert f["inter_tx_time_variance"] == 0.0


def test_inter_tx_time_variance_regular_spacing_is_zero():
    base = _AIRDROP_TS
    txs = [
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base), "gasPrice": "20000000000"},
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base + 60), "gasPrice": "20000000000"},
        {"from": _ADDR, "to": "0xother", "value": "0", "isError": "0",
         "timeStamp": str(base + 120), "gasPrice": "20000000000"},
    ]
    f = compute_airdrop_features(_ADDR, txs, [], _CONTRACT, _AIRDROP_DATE, set())
    assert f["inter_tx_time_variance"] == 0.0


def test_unique_counterparty_count():
    f = compute_airdrop_features(_ADDR, _TXS_SYBIL, _TOKEN_TXS_SYBIL, _CONTRACT, _AIRDROP_DATE, set())
    # counterparties: 0xfunder, 0xcontract, 0xexchange = 3
    assert f["unique_counterparty_count"] == 3.0


def test_extractor_calls_etherscan_and_delegates():
    client = MagicMock()
    client.get_tx_list.return_value = _TXS_SYBIL
    client.get_token_transfers.return_value = _TOKEN_TXS_SYBIL
    extractor = AirdropFeatureExtractor(client, _CONTRACT, _AIRDROP_DATE)
    features = extractor.extract(_ADDR)
    client.get_tx_list.assert_called_once_with(_ADDR)
    client.get_token_transfers.assert_called_once_with(_ADDR)
    assert "wallet_age_days" in features


def test_empty_txs_returns_zero_features():
    f = compute_airdrop_features(_ADDR, [], [], _CONTRACT, _AIRDROP_DATE, set())
    assert all(v == 0.0 for v in f.values())
