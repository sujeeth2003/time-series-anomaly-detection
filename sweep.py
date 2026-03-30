"""Repeat the comparison over several random seeds and report mean +/- std (one seed with 16 events is too noisy to trust).

    python sweep.py [--seeds 8] [--budget 6]
"""
import argparse

import numpy as np

from tsad.detectors import IsolationForestDetector, SupervisedDetector, zscore_baseline
from tsad.evaluate import evaluate, threshold_for_fpr
from tsad.features import window_features
from tsad.synth import make_stream

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--budget", type=float, default=6.0)
a = ap.parse_args()

