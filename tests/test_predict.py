import numpy as np
import pytest
from unittest.mock import MagicMock
from blockchain_ai.predict import predict_address, LABEL_ENCODER


def _mock_model(probas):
    m = MagicMock()
    m.predict_proba.return_value = np.array([probas])
    return m


def test_predict_address_returns_highest_label():
    result = predict_address(
        features={"tx_count": 10.0, "account_age_days": 30.0},
        model=_mock_model([0.05, 0.85, 0.10]),
        feature_cols=["tx_count", "account_age_days"],
        threshold=0.5,
    )
    assert result["label"] == "scammer"
    assert result["probabilities"]["scammer"] == pytest.approx(0.85, abs=0.001)


def test_predict_address_returns_unknown_below_threshold():
    result = predict_address(
        features={"tx_count": 5.0, "account_age_days": 10.0},
        model=_mock_model([0.35, 0.35, 0.30]),
        feature_cols=["tx_count", "account_age_days"],
        threshold=0.5,
    )
    assert result["label"] == "unknown"


def test_predict_address_probabilities_have_three_keys():
    result = predict_address(
        features={"tx_count": 5.0},
        model=_mock_model([0.1, 0.7, 0.2]),
        feature_cols=["tx_count"],
        threshold=0.5,
    )
    assert set(result["probabilities"].keys()) == {"sanctioned", "scammer", "phishing"}


def test_label_encoder_covers_all_classes():
    assert set(LABEL_ENCODER.values()) == {"sanctioned", "scammer", "phishing"}
