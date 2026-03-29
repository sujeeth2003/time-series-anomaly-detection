"""Synthetic industrial sensor streams with labelled anomaly events.

Channels behave like a small machine: a slow operating cycle, a vibration-like oscillation, correlated
noise. Four anomaly types are injected, each with a ground-truth [start, end) window:
  spike        short large excursions (sensor glitch)
  drift        slow ramp away from the baseline (wear, fouling)
  stuck        the channel freezes at a constant value (dead sensor)
  variance     noise level jumps (loose mounting, bearing damage)
"""
import numpy as np

KINDS = ("spike", "drift", "stuck", "variance")


def make_stream(n=20000, channels=4, n_events=12, seed=0, fs=10.0):
    """Returns (X[n, channels], labels[n] in {0,1}, events[list of (start, end, kind, channel)])."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    X = np.empty((n, channels))
    for c in range(channels):
        cycle = np.sin(2 * np.pi * t / (60 + 7 * c) + c)
        vib = 0.4 * np.sin(2 * np.pi * (1.3 + 0.4 * c) * t)
        X[:, c] = 10 + 2 * cycle + vib + rng.normal(0, 0.25, n)
    y = np.zeros(n, dtype=int)
    events = []
    # place events on a grid so they never overlap and always leave normal data between them
    slots = np.linspace(1500, n - 1500, n_events).astype(int)
    for i, s in enumerate(slots):
        kind, ch = KINDS[i % len(KINDS)], int(rng.integers(channels))
        length = {"spike": 6, "drift": 400, "stuck": 250, "variance": 300}[kind]
        e = min(n, s + length)
        seg = slice(s, e)
        if kind == "spike":
            X[seg, ch] += rng.choice([-1, 1]) * rng.uniform(3, 5)
        elif kind == "drift":
            X[seg, ch] += np.linspace(0, rng.uniform(2.5, 4.0), e - s)
            X[e:, ch] += 0  # drift ends; sensor recovers after maintenance
        elif kind == "stuck":
            X[seg, ch] = X[s, ch]
        else:
            X[seg, ch] += rng.normal(0, 1.2, e - s)
        y[seg] = 1
        events.append((int(s), int(e), kind, ch))
    return X, y, events
