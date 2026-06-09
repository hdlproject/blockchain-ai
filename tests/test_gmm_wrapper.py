import numpy as np
import pytest
from blockchain_ai.model.gmm_wrapper import GMMWrapper

_FEATURE_COLS = [
    "wallet_age_days",
    "tx_count_pre_airdrop",
    "token_type_diversity",
    "claim_to_withdraw_hours",
    "gas_source_shared",
    "inter_tx_time_variance",
    "unique_counterparty_count",
]

_RNG = np.random.default_rng(42)


def _synthetic_data(n: int = 200) -> np.ndarray:
    """Generate 4 clusters of synthetic wallet features."""
    genuine = _RNG.normal([365, 50, 10, 720, 0, 50000, 30], [50, 20, 5, 200, 0.1, 10000, 10], (n // 4, 7))
    casual  = _RNG.normal([180, 20, 5, 48, 0, 10000, 15], [30, 10, 3, 20, 0.1, 5000, 5], (n // 4, 7))
    light   = _RNG.normal([30, 3, 2, 4, 0.3, 1000, 5], [10, 2, 1, 2, 0.2, 500, 3], (n // 4, 7))
    heavy   = _RNG.normal([10, 1, 1, 0.5, 0.9, 100, 2], [5, 1, 0.5, 0.3, 0.1, 50, 1], (n // 4, 7))
    return np.vstack([genuine, casual, light, heavy]).clip(0)


def test_fit_returns_self():
    wrapper = GMMWrapper()
    result = wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    assert result is wrapper


def test_bic_scores_has_seven_entries():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    assert len(wrapper.bic_scores) == 7
    ks = [entry["k"] for entry in wrapper.bic_scores]
    assert ks == list(range(2, 9))


def test_bic_scores_are_floats():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    for entry in wrapper.bic_scores:
        assert isinstance(entry["bic"], float)


def test_farmer_score_in_unit_interval():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    genuine_features = {
        "wallet_age_days": 400.0, "tx_count_pre_airdrop": 60.0,
        "token_type_diversity": 12.0, "claim_to_withdraw_hours": 800.0,
        "gas_source_shared": 0.0, "inter_tx_time_variance": 60000.0,
        "unique_counterparty_count": 35.0,
    }
    score = wrapper.score_wallet(genuine_features, _FEATURE_COLS)
    assert 0.0 <= score["farmer_score"] <= 1.0


def test_heavy_farmer_scores_higher_than_genuine():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    farmer_features = {
        "wallet_age_days": 5.0, "tx_count_pre_airdrop": 0.0,
        "token_type_diversity": 1.0, "claim_to_withdraw_hours": 0.2,
        "gas_source_shared": 1.0, "inter_tx_time_variance": 50.0,
        "unique_counterparty_count": 1.0,
    }
    genuine_features = {
        "wallet_age_days": 400.0, "tx_count_pre_airdrop": 60.0,
        "token_type_diversity": 12.0, "claim_to_withdraw_hours": 800.0,
        "gas_source_shared": 0.0, "inter_tx_time_variance": 60000.0,
        "unique_counterparty_count": 35.0,
    }
    farmer_score = wrapper.score_wallet(farmer_features, _FEATURE_COLS)["farmer_score"]
    genuine_score = wrapper.score_wallet(genuine_features, _FEATURE_COLS)["farmer_score"]
    assert farmer_score > genuine_score


def test_priority_tier_normal():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    wrapper._farmer_cluster_indices = []
    features = {col: 1.0 for col in _FEATURE_COLS}
    result = wrapper.score_wallet(features, _FEATURE_COLS)
    assert result["priority_tier"] == "normal"
    assert result["farmer_score"] == 0.0


def test_priority_tier_deprioritize():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    wrapper._farmer_cluster_indices = list(range(wrapper.n_components))
    features = {col: 1.0 for col in _FEATURE_COLS}
    result = wrapper.score_wallet(features, _FEATURE_COLS)
    assert result["priority_tier"] == "deprioritize"
    assert result["farmer_score"] == pytest.approx(1.0, abs=0.01)


def test_score_wallet_result_contains_bic_scores():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS)
    features = {col: 1.0 for col in _FEATURE_COLS}
    result = wrapper.score_wallet(features, _FEATURE_COLS)
    assert "bic_scores" in result
    assert len(result["bic_scores"]) == 7


def test_funding_address_set_stored_on_fit():
    wrapper = GMMWrapper()
    wrapper.fit(_synthetic_data(), _FEATURE_COLS, funding_address_set={"0xABC", "0xDEF"})
    assert "0xabc" in wrapper.funding_address_set
    assert "0xdef" in wrapper.funding_address_set


def test_farmer_score_before_fit_raises():
    wrapper = GMMWrapper()
    with pytest.raises(RuntimeError, match="fit"):
        wrapper.farmer_score(np.ones(7))
