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

