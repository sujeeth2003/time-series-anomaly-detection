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

