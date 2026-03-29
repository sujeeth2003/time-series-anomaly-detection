"""Event-level evaluation. Accuracy per sample is misleading for rare events; what matters operationally is:
  * did we catch the event at all (recall over events)
  * how LONG after the anomaly began did we alarm (detection delay)
  * how many false alarms per hour of normal operation (false-positive rate)
"""
import numpy as np


