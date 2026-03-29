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


