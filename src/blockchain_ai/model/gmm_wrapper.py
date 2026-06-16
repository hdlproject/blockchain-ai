import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMWrapper:
    def __init__(self, n_components: int = 4, covariance_type: str = "full", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self._gmm: GaussianMixture | None = None
        self._scaler: StandardScaler | None = None
        self._feature_cols: list[str] = []
        self._farmer_cluster_indices: list[int] = []
        self._bic_scores: list[dict] = []

    def fit(self, X: np.ndarray, feature_cols: list[str]) -> "GMMWrapper":
        X = np.asarray(X, dtype=float)
        self._feature_cols = feature_cols
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._bic_scores = self._compute_bic(X_scaled)
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        self._gmm.fit(X_scaled)
        self._farmer_cluster_indices = self._identify_farmer_clusters()
        return self

    def _compute_bic(self, X_scaled: np.ndarray) -> list[dict]:
        scores = []
        for k in range(2, 9):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gmm.fit(X_scaled)
            scores.append({"k": k, "bic": float(gmm.bic(X_scaled))})
        return scores

    def _identify_farmer_clusters(self) -> list[int]:
        if self._gmm is None or self._scaler is None:
            raise RuntimeError("Call fit() first")
        means = self._scaler.inverse_transform(self._gmm.means_)
        feature_range = means.max(axis=0) - means.min(axis=0) + 1e-10
        normalized = (means - means.min(axis=0)) / feature_range
        farmer_signals = {
            "wallet_age_days": "low",
            "tx_count_before_first_inflow": "low",
            "inflow_to_outflow_hours": "low",
            "shared_funder_score": "high",
            "unique_counterparty_count": "low",
        }
        farmer_proxy = np.zeros(self.n_components)
        for feature, direction in farmer_signals.items():
            if feature in self._feature_cols:
                idx = self._feature_cols.index(feature)
                if direction == "low":
                    farmer_proxy += 1 - normalized[:, idx]
                else:
                    farmer_proxy += normalized[:, idx]
        return np.argsort(farmer_proxy)[-2:].tolist()

    def farmer_score(self, x: np.ndarray) -> float:
        if self._gmm is None or self._scaler is None:
            raise RuntimeError("Call fit() first")
        x_scaled = self._scaler.transform(np.asarray(x, dtype=float).reshape(1, -1))
        proba = self._gmm.predict_proba(x_scaled)[0]
        return float(sum(proba[i] for i in self._farmer_cluster_indices))

    def score_wallet(self, features: dict, feature_cols: list[str]) -> dict:
        if self._gmm is None or self._scaler is None:
            raise RuntimeError("Call fit() first")
        x = np.array([features[col] for col in feature_cols], dtype=float)
        score = self.farmer_score(x)
        if score < 0.4:
            tier = "normal"
        elif score <= 0.8:
            tier = "watch"
        else:
            tier = "deprioritize"
        return {
            "farmer_score": round(score, 4),
            "priority_tier": tier,
            "bic_scores": self._bic_scores,
        }

    @property
    def bic_scores(self) -> list[dict]:
        return self._bic_scores
