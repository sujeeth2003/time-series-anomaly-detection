"""Event-level evaluation. Accuracy per sample is misleading for rare events; what matters operationally is:
  * did we catch the event at all (recall over events)
  * how LONG after the anomaly began did we alarm (detection delay)
  * how many false alarms per hour of normal operation (false-positive rate)
"""
import numpy as np


def alarms_from_scores(scores, threshold, min_gap=50):
    """Turn a score stream into discrete alarm times: first sample above threshold, then a refractory gap."""
    idx = np.flatnonzero(scores > threshold)
    out, last = [], -10 ** 9
    for i in idx:
        if i - last >= min_gap:
            out.append(int(i))
        last = i
    return out


def evaluate(scores, events, threshold, n, fs=10.0, min_gap=50, grace=100):
    """events: list of (start, end, kind, channel). An alarm is a hit if it falls in [start, end + grace)."""
    alarms = alarms_from_scores(scores, threshold, min_gap)
    hit_delay, caught = [], 0
    used = set()
    for (s, e, kind, ch) in events:
        first = next((a for a in alarms if s <= a < e + grace), None)
        if first is not None:
            caught += 1
            hit_delay.append((first - s) / fs)
            used.update(a for a in alarms if s <= a < e + grace)
    false_alarms = [a for a in alarms if a not in used]
    normal_seconds = (n - sum(e - s for s, e, _, _ in events)) / fs
    return {
        "event_recall": caught / max(len(events), 1),
        "false_alarms": len(false_alarms),
        "false_alarms_per_hour": len(false_alarms) / (normal_seconds / 3600),
        "median_delay_s": float(np.median(hit_delay)) if hit_delay else float("nan"),
        "mean_delay_s": float(np.mean(hit_delay)) if hit_delay else float("nan"),
    }


def threshold_for_fpr(scores_normal, target_false_alarms_per_hour, fs=10.0, min_gap=50):
    """Pick the smallest threshold whose false-alarm rate on known-normal data stays under the target."""
    hours = len(scores_normal) / fs / 3600
    for thr in np.quantile(scores_normal, np.linspace(0.90, 0.99999, 400)):
        if len(alarms_from_scores(scores_normal, thr, min_gap)) / hours <= target_false_alarms_per_hour:
            return float(thr)
    return float(scores_normal.max())
