import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class DBSCANWrapper:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._scaler: StandardScaler | None = None
        self._core_samples: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> "DBSCANWrapper":
        X = np.asarray(X, dtype=np.float32)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        db.fit(X_scaled)
        if len(db.core_sample_indices_) > 0:
            self._core_samples = X_scaled[db.core_sample_indices_]
        else:
            self._core_samples = X_scaled
        self._nn = NearestNeighbors(n_neighbors=1)
        self._nn.fit(self._core_samples)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None or self._scaler is None:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float32))
        distances, _ = self._nn.kneighbors(X_scaled)
        return distances[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        return np.where(scores <= self.eps, 0, -1).astype(np.int32)
