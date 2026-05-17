import numpy as np
import pandas as pd
import pytest
import joblib
from blockchain_ai.model.lstm import LSTMWrapper

_N = 60
_FEATURES = 3
_SEQ_LEN = 5


def _data(n=_N, features=_FEATURES, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, features)).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1]
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(features)]), pd.Series(y)


def test_fit_predict_returns_finite_array():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=3, hidden_size=8, num_layers=1)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert np.isfinite(preds).all()


def test_predict_shape_matches_input_length():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    assert model.predict(X).shape == (len(X),)


def test_predict_zero_padding_for_short_input():
    """predict() on fewer rows than seq_len must still return that many predictions."""
    X, y = _data(n=40)
    model = LSTMWrapper(sequence_length=10, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    short = X.iloc[:3]
    preds = model.predict(short)
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()


def test_predict_accepts_numpy_array():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    preds = model.predict(X.values)
    assert preds.shape == (len(X),)


def test_feature_cols_stored_after_fit():
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=2, hidden_size=4, num_layers=1)
    model.fit(X, y)
    assert model.feature_cols == ["f0", "f1", "f2"]


def test_early_stopping_does_not_crash():
    """Training completes without error even if early stopping fires."""
    X, y = _data(n=100)
    model = LSTMWrapper(
        sequence_length=_SEQ_LEN, epochs=50, hidden_size=4, num_layers=1, dropout=0.0
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_joblib_round_trip_preserves_predictions(tmp_path):
    X, y = _data()
    model = LSTMWrapper(sequence_length=_SEQ_LEN, epochs=3, hidden_size=8, num_layers=1)
    model.fit(X, y)
    before = model.predict(X)

    path = tmp_path / "lstm.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    after = loaded.predict(X)

    np.testing.assert_array_almost_equal(before, after, decimal=5)


def test_sequence_length_attribute_accessible():
    model = LSTMWrapper(sequence_length=15)
    assert model.sequence_length == 15
