"""Window features per time step, computed causally (only the past W samples: usable online)."""
import numpy as np


def _rolling(a, w, fn):
    """fn over the trailing window of length w for every step (pads the start by edge values)."""
    pad = np.concatenate([np.repeat(a[:1], w - 1, axis=0), a], axis=0)
    idx = np.arange(len(a))[:, None] + np.arange(w)[None, :]
    return fn(pad[idx], axis=1)


