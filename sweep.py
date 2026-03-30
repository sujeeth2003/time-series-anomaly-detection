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

res = {}
for seed in range(a.seeds):
    XA, yA, _ = make_stream(seed=10 * seed, n=24000, n_events=16)
    XB, _, _ = make_stream(seed=10 * seed + 1, n=20000, n_events=0)
    XC, yC, evC = make_stream(seed=10 * seed + 2, n=24000, n_events=16)
    FA, _ = window_features(XA); FB, _ = window_features(XB); FC, _ = window_features(XC)
    iso = IsolationForestDetector(seed).fit(FA[yA == 0])
    sup = SupervisedDetector(seed).fit(FA, yA)
    for name, sb, sc in (("rolling z-score", zscore_baseline(XB), zscore_baseline(XC)),
                         ("isolation forest", iso.score(FB), iso.score(FC)),
                         ("gradient boosting (supervised)", sup.score(FB), sup.score(FC))):
        r = evaluate(sc, evC, threshold_for_fpr(sb, a.budget), len(XC))
        res.setdefault(name, []).append((r["event_recall"], r["false_alarms_per_hour"], r["median_delay_s"]))

print(f"{a.seeds} seeds, false-alarm budget {a.budget}/hour on normal data (mean +/- std over seeds)\n")
print(f"{'detector':<32}{'event recall':>16}{'FA / hour':>16}{'median delay (s)':>20}")
for name, rows in res.items():
    m, s = np.nanmean(rows, axis=0), np.nanstd(rows, axis=0)
    print(f"{name:<32}{m[0]:>10.2f} +/- {s[0]:.2f}{m[1]:>10.1f} +/- {s[1]:.1f}{m[2]:>12.1f} +/- {s[2]:.1f}")
