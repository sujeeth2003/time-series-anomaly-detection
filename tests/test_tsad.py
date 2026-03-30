import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tsad.evaluate import alarms_from_scores, evaluate, threshold_for_fpr  # noqa: E402
from tsad.features import window_features  # noqa: E402
from tsad.synth import make_stream  # noqa: E402


