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

    detectors = {
        "rolling z-score": (zscore_baseline(XB), zscore_baseline(XC)),
        "isolation forest": (iso.score(FB), iso.score(FC)),
        "gradient boosting (supervised)": (sup.score(FB), sup.score(FC)),
    }
    print(f"false-alarm budget: {a.budget}/hour on normal data;  test stream: {len(XC)} samples, {len(evC)} events\n")
    print(f"{'detector':<32}{'event recall':>13}{'false alarms':>14}{'FA / hour':>11}{'median delay (s)':>18}")
    for name, (sb, sc) in detectors.items():
        thr = threshold_for_fpr(sb, a.budget)
        r = evaluate(sc, evC, thr, len(XC))
        print(f"{name:<32}{r['event_recall']:>13.2f}{r['false_alarms']:>14d}{r['false_alarms_per_hour']:>11.1f}{r['median_delay_s']:>18.1f}")

    print("\nper anomaly type (event recall / median delay in s), supervised vs z-score:")
    for kind in ("spike", "drift", "stuck", "variance"):
        ev = [e for e in evC if e[2] == kind]
        row = []
        for name in ("rolling z-score", "gradient boosting (supervised)"):
            sb, sc = detectors[name]
            r = evaluate(sc, ev, threshold_for_fpr(sb, a.budget), len(XC))
            row.append(f"{r['event_recall']:.2f} / {r['median_delay_s']:.1f}s")
        print(f"  {kind:<9} z-score {row[0]:<14} supervised {row[1]}")


if __name__ == "__main__":
    main()
