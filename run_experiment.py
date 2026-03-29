"""Compare the three detectors on synthetic sensor streams at the SAME false-alarm budget.

    python run_experiment.py [--budget 6]      # allowed false alarms per hour on normal data

Method: train on stream A (labelled), calibrate every detector's threshold on a separate normal-only stream B
so all three have the same false-alarm rate budget, then test on unseen stream C. Comparing detectors at
equal false-alarm rate is the honest way; comparing at each one's default threshold is not.
"""
import argparse

import numpy as np

from tsad.detectors import IsolationForestDetector, SupervisedDetector, zscore_baseline
from tsad.evaluate import evaluate, threshold_for_fpr
from tsad.features import window_features
from tsad.synth import make_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=6.0, help="allowed false alarms per hour on normal data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    XA, yA, evA = make_stream(seed=a.seed, n=24000, n_events=16)             # train (labelled)
    XB, _, _ = make_stream(seed=a.seed + 1, n=20000, n_events=0)            # calibration: normal only
    XC, yC, evC = make_stream(seed=a.seed + 2, n=24000, n_events=16)        # test (unseen)
    FA, _ = window_features(XA); FB, _ = window_features(XB); FC, _ = window_features(XC)

    iso = IsolationForestDetector(a.seed).fit(FA[yA == 0])                  # trained on normal-looking windows only
    sup = SupervisedDetector(a.seed).fit(FA, yA)

