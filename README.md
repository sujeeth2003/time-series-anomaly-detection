# Time-Series Anomaly Detection on Industrial Sensors

Unsupervised and supervised detectors for multi-channel sensor streams, judged the way an operator would judge them: **how many real events did it catch, how long after they started, and how many false alarms per hour** (not per-sample accuracy, which is meaningless when anomalies are rare).

Background: this comes from my work on 50+ channel, 100 kHz telemetry from satellite thruster tests, where a detector is only useful if it alarms early and rarely cries wolf. **The data here is synthetic** (the real telemetry is not mine to publish), so the numbers below demonstrate the method, not performance on real machines.

## Pipeline
| Step | Code |
|---|---|
| Synthetic streams: operating cycle + vibration + noise, with labelled `spike`, `drift`, `stuck` (dead sensor) and `variance` events | `tsad/synth.py` |
| Causal window features per channel: mean, std, range, slope, first-difference std, high-frequency spectral energy (FFT) | `tsad/features.py` |
| Detectors: rolling z-score baseline, **Isolation Forest** (unsupervised, trained on normal-looking windows), **gradient boosting** (supervised) | `tsad/detectors.py` |
| Event-level metrics: event recall, false alarms per hour, detection delay; threshold chosen for a fixed false-alarm budget | `tsad/evaluate.py` |

Design choices that matter:
- **Equal false-alarm budget.** Every detector's threshold is calibrated on a separate normal-only stream to allow the same alarms per hour. Comparing detectors at their default thresholds is not a fair comparison.
- **Causal features.** A test checks that changing the future never changes earlier feature rows, so the same code is valid online.
- **Train / calibrate / test on three different streams.**

## Results (synthetic, 8 seeds, budget = 6 false alarms/hour, mean +/- std)
```
detector                            event recall       FA / hour    median delay (s)
rolling z-score                       0.77 +/- 0.07       7.1 +/- 6.6        12.9 +/- 8.7
isolation forest                      0.98 +/- 0.04       4.0 +/- 3.7         3.4 +/- 1.2
gradient boosting (supervised)        0.62 +/- 0.09       5.8 +/- 3.5         5.7 +/- 3.2
```
Reading it honestly:
- **Isolation Forest on window features wins** here: it catches nearly every event, earliest, with the fewest false alarms, without needing any labels.
- The **supervised model does worse**, because it only sees 16 labelled events to learn from and the labels cover long stretches of which only some windows look abnormal. Supervised detection would need far more labelled history; this is a data-volume result, not a general claim about supervised models.
- The **z-score baseline** is strong on spikes and variance changes (near-instant) but is poor on slow **drift**, because a rolling baseline adapts to the drift. That is exactly why drift needs window/slope features.
- False-alarm rates vary a lot from seed to seed (std about as large as the mean): treat any single-seed number, including the one in `run_experiment.py`, as an example.

