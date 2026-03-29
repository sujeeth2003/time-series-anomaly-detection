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


