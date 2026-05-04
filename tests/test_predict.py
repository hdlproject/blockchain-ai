import joblib
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from unittest.mock import MagicMock
from blockchain_ai.predict import run_inference, predict_address, LABEL_ENCODER


def test_run_inference_returns_predictions(tmp_path):
    csv_path = tmp_path / "input.csv"
    model_path = tmp_path / "model.joblib"

    train_df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    model = LinearRegression()
    model.fit(train_df[["feature1"]], train_df["target"])
    joblib.dump(model, model_path)

    input_df = pd.DataFrame({"feature1": [4.0, 5.0]})
    input_df.to_csv(csv_path, index=False)

    predictions = run_inference(str(csv_path), str(model_path))

    assert len(predictions) == 2
    assert np.isclose(predictions[0], 8.0, atol=0.1)
    assert np.isclose(predictions[1], 10.0, atol=0.1)


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
