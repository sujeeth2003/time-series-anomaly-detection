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


