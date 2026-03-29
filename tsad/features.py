"""Window features per time step, computed causally (only the past W samples: usable online)."""
import numpy as np


def _rolling(a, w, fn):
    """fn over the trailing window of length w for every step (pads the start by edge values)."""
    pad = np.concatenate([np.repeat(a[:1], w - 1, axis=0), a], axis=0)
    idx = np.arange(len(a))[:, None] + np.arange(w)[None, :]
    return fn(pad[idx], axis=1)


def window_features(X, w=50):
    """For each channel: rolling mean, std, min-to-max range, slope, plus first-difference std (spike/stuck sensitive)
    and the fraction of spectral energy above 0.5 Hz-equivalent (variance/vibration changes)."""
    n, ch = X.shape
    feats, names = [], []
    t = np.arange(w) - (w - 1) / 2
    for c in range(ch):
        a = X[:, c]
        mean = _rolling(a, w, np.mean)
        std = _rolling(a, w, np.std)
        rng_ = _rolling(a, w, np.max) - _rolling(a, w, np.min)
        d = np.diff(a, prepend=a[0])
        dstd = _rolling(d, w, np.std)
        win = np.lib.stride_tricks.sliding_window_view(np.concatenate([np.repeat(a[:1], w - 1), a]), w)
        slope = (win * t).sum(axis=1) / (t ** 2).sum()
        spec = np.abs(np.fft.rfft(win - win.mean(axis=1, keepdims=True), axis=1)) ** 2
        hf = spec[:, w // 4:].sum(axis=1) / (spec.sum(axis=1) + 1e-9)
        for nm, v in (("mean", mean), ("std", std), ("range", rng_), ("dstd", dstd), ("slope", slope), ("hf", hf)):
            feats.append(v); names.append(f"ch{c}_{nm}")
    return np.column_stack(feats), names
