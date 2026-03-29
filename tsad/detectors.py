"""Three detectors of increasing sophistication. Each returns a per-step anomaly SCORE (higher = more anomalous)."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import _rolling


