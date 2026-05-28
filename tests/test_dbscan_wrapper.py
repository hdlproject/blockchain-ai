import numpy as np
import pytest
from blockchain_ai.model.dbscan_wrapper import DBSCANWrapper


def _make_data():
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(80, 2))
    cluster_b = rng.normal(loc=[5.0, 5.0], scale=0.3, size=(80, 2))
    outlier = np.array([[20.0, 20.0]])
    return np.vstack([cluster_a, cluster_b, outlier])


def test_fit_returns_self():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    result = model.fit(X)
    assert result is model


def test_predict_labels_outlier_as_anomaly():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    model.fit(X)
    outlier = np.array([[20.0, 20.0]])
    labels = model.predict(outlier)
    assert labels[0] == -1


def test_predict_labels_cluster_point_as_normal():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    model.fit(X)
    normal = np.array([[0.0, 0.0]])
    labels = model.predict(normal)
    assert labels[0] == 0


def test_anomaly_score_outlier_higher_than_normal():
    X = _make_data()
    model = DBSCANWrapper(eps=0.5, min_samples=5)
    model.fit(X)
    normal_score = model.anomaly_score(np.array([[0.0, 0.0]]))[0]
    outlier_score = model.anomaly_score(np.array([[20.0, 20.0]]))[0]
    assert outlier_score > normal_score


def test_predict_before_fit_raises():
    model = DBSCANWrapper()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(np.array([[1.0, 2.0]]))


def test_all_noise_fallback():
    # When all points are noise, should still not raise
    X = np.eye(10) * 100  # 10 isolated points, no density
    model = DBSCANWrapper(eps=0.1, min_samples=5)
    model.fit(X)
    scores = model.anomaly_score(X)
    assert scores.shape == (10,)
