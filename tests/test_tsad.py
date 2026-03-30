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

    def test_alarm_refractory_gap(self):
        s = np.zeros(500); s[[10, 11, 12, 200, 201]] = 5
        self.assertEqual(alarms_from_scores(s, 1, min_gap=50), [10, 200])

    def test_evaluate_counts_hits_delay_and_false_alarms(self):
        n = 3600 * 10                                  # 1 hour at 10 Hz
        events = [(1000, 1100, "drift", 0), (5000, 5100, "stuck", 1)]
        s = np.zeros(n); s[1030] = 9; s[20000] = 9    # hit event 1 after 3.0 s; miss event 2; one false alarm
        r = evaluate(s, events, 1.0, n)
        self.assertEqual(r["event_recall"], 0.5)
        self.assertEqual(r["false_alarms"], 1)
        self.assertAlmostEqual(r["median_delay_s"], 3.0)

    def test_threshold_respects_false_alarm_budget(self):
        rng = np.random.default_rng(0)
        normal = rng.normal(size=72000)               # 2 hours
        thr = threshold_for_fpr(normal, 5.0)
        per_hour = len(alarms_from_scores(normal, thr)) / 2
        self.assertLessEqual(per_hour, 5.0)


if __name__ == "__main__":
    unittest.main()
