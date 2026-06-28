import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


def _elbow_eps(X_scaled: np.ndarray, min_samples: int) -> float:
    """Find eps from the k-distance elbow: the point farthest from the diagonal
    of the sorted k-distance curve, normalised to [0,1] x [0,1]."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    k_distances = np.sort(nn.kneighbors(X_scaled)[0][:, -1])
    n = len(k_distances)
    x = np.linspace(0, 1, n)
    d_min, d_max = k_distances[0], k_distances[-1]
    y = (k_distances - d_min) / (d_max - d_min + 1e-10)
    # perpendicular distance from each point to the line y=x
    elbow_idx = int(np.argmax(np.abs(x - y)))
    return float(k_distances[elbow_idx])


class DBSCANWrapper:
    """
    Unsupervised anomaly detector based on DBSCAN.

    Core assumption: training data is dominated by normal transactions.
    Dense regions in feature space = normal behaviour.
    Sparse regions (far from any core point) = anomaly.

    This assumption breaks if:
      - Training data contains many anomalies (they form their own dense clusters
        and become "normal" by definition).
      - Anomaly ratio in training output is 30%+ — sign of bad training data.
      - Attackers mimic normal tx patterns — blend into clusters undetected.

    Anomaly scoring at inference:
      score = distance to nearest core point from training.
      score > eps  → farther than ε from any dense region → anomaly.
      score ≤ eps  → within ε of a known cluster → normal.

    ε serves double duty: cluster membership during training AND
    the decision threshold at inference.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, auto_eps: bool = False):
        self.eps = eps
        self.min_samples = min_samples
        self.auto_eps = auto_eps
        self._scaler: RobustScaler | None = None
        self._core_samples: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def fit(self, X: np.ndarray) -> "DBSCANWrapper":
        X = np.asarray(X, dtype=np.float32)
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.auto_eps:
            self.eps = _elbow_eps(X_scaled, self.min_samples)
            print(f"[auto_eps] elbow detected at eps={self.eps:.4f}")

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
            raise RuntimeError("Call fit() before using the model.")
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float32))
        distances, _ = self._nn.kneighbors(X_scaled)
        return distances[:, 0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        return np.where(scores <= self.eps, 0, -1).astype(np.int32)
