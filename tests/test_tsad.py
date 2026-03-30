import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tsad.evaluate import alarms_from_scores, evaluate, threshold_for_fpr  # noqa: E402
from tsad.features import window_features  # noqa: E402
from tsad.synth import make_stream  # noqa: E402


class TSADTests(unittest.TestCase):
    def test_synth_labels_match_events(self):
        X, y, ev = make_stream(n=8000, n_events=8, seed=3)
        self.assertEqual(X.shape, (8000, 4))
        self.assertEqual(int(y.sum()), sum(e - s for s, e, _, _ in ev))
        spans = sorted((s, e) for s, e, _, _ in ev)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "events overlap")

    def test_features_are_causal(self):
        # changing the FUTURE must not change earlier feature rows
        X, _, _ = make_stream(n=1000, n_events=0, seed=1)
        F1, _ = window_features(X)
        X2 = X.copy(); X2[600:] += 100
        F2, _ = window_features(X2)
        np.testing.assert_allclose(F1[:600], F2[:600])

