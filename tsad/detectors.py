"""Three detectors of increasing sophistication. Each returns a per-step anomaly SCORE (higher = more anomalous)."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import _rolling


def zscore_baseline(X, w=200):
    """Rolling z-score of each channel against its own recent past; score = max over channels."""
    scores = []
    for c in range(X.shape[1]):
        a = X[:, c]
        mu = _rolling(a, w, np.mean)
        sd = _rolling(a, w, np.std) + 1e-6
        prev_mu = np.concatenate([[mu[0]], mu[:-1]])
        prev_sd = np.concatenate([[sd[0]], sd[:-1]])
        scores.append(np.abs(a - prev_mu) / prev_sd)
    return np.max(scores, axis=0)


class IsolationForestDetector:
    """Unsupervised: fit on a stretch assumed to be mostly normal, score = -decision_function."""

    def __init__(self, seed=0):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)

    def fit(self, F):
        self.model.fit(self.scaler.fit_transform(F))
        return self

    def score(self, F):
        return -self.model.decision_function(self.scaler.transform(F))


class SupervisedDetector:
    """Gradient boosting on window features with labelled history; score = P(anomaly)."""

    def __init__(self, seed=0):
        self.model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed)

    def fit(self, F, y):
        self.model.fit(F, y)
        return self

    def score(self, F):
        return self.model.predict_proba(F)[:, 1]
