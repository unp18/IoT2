"""
Sudden-temperature-spike ("fire") detection.

  extract_features()  turns the last N temperature readings into numbers describing the *shape* of the trend
  FireModel           a trained logistic-regression model, loaded from fire_model.json (pure Python inference)
  SpikeDetector       stateful wrapper used by the server: rolling window + model + confirmation + safety gate

This module is imported by BOTH train_fire_model.py and main.py, so training and live inference always
compute exactly the same features. It has no third-party dependencies.
"""

import json
import math
from collections import deque
from pathlib import Path
from typing import Optional, Sequence

FEATURES = [
    "rise",          # last - first reading
    "slope",         # least-squares slope, degC per reading
    "last3_rise",    # rise over the most recent 3 steps (is it still climbing?)
    "max_step",      # biggest single step up
    "min_step",      # biggest single step down
    "step_std",      # spread of step sizes
    "frac_nonneg",   # fraction of steps that did not go down
    "accel",         # slope of 2nd half minus slope of 1st half (accelerating?)
    "range",         # max - min
    "peak_gap",      # max - last (0 when the last reading is the peak)
    "resid_std",     # deviation from a straight line (noisy vs clean ramp)
    "jump_share",    # biggest up-step / total rise (near 1.0 = one glitchy jump, not a trend)
]


def extract_features(w: Sequence[float]) -> list[float]:
    n = len(w)
    if n < 4:
        raise ValueError("need at least 4 readings")
    d = [b - a for a, b in zip(w, w[1:])]
    rise = w[-1] - w[0]

    xm, ym = (n - 1) / 2.0, sum(w) / n
    sxx = sum((i - xm) ** 2 for i in range(n))
    sxy = sum((i - xm) * (y - ym) for i, y in enumerate(w))
    slope = sxy / sxx
    icpt = ym - slope * xm
    resid_std = math.sqrt(sum((y - (icpt + slope * i)) ** 2 for i, y in enumerate(w)) / n)

    mean_d = rise / (n - 1)
    step_std = math.sqrt(sum((x - mean_d) ** 2 for x in d) / len(d))
    half = n // 2
    accel = (w[-1] - w[half]) / (n - 1 - half) - (w[half] - w[0]) / half

    return [
        rise,
        slope,
        w[-1] - w[-4],
        max(d),
        min(d),
        step_std,
        sum(1 for x in d if x >= 0) / len(d),
        accel,
        max(w) - min(w),
        max(w) - w[-1],
        resid_std,
        min(2.0, max(0.0, max(d)) / max(abs(rise), 0.5)),
    ]


class FireModel:
    """Logistic regression exported to JSON by train_fire_model.py."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.window: int = int(spec["window"])
        self.threshold: float = float(spec["threshold"])
        self.mean = spec["mean"]
        self.scale = spec["scale"]
        self.coef = spec["coef"]
        self.intercept = float(spec["intercept"])
        if spec["features"] != FEATURES:
            raise ValueError("fire_model.json was trained with different features: retrain the model")

    @classmethod
    def load(cls, path) -> "FireModel":
        return cls(json.loads(Path(path).read_text()))

    def predict_proba(self, window: Sequence[float]) -> float:
        f = extract_features(window)
        z = self.intercept + sum(c * (x - m) / s for c, x, m, s in zip(self.coef, f, self.mean, self.scale))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def info(self) -> dict:
        keys = ("model_type", "window", "threshold", "trained_at", "dataset", "metrics", "comparison", "features")
        return {k: self.spec.get(k) for k in keys}


class SpikeDetector:
    """
    Feed it one temperature reading at a time with update().

    A reading triggers when: the window is full, the model score >= threshold, the temperature is at
    least `min_temp` (absolute safety gate, so a cold room warming up never alarms), and this has held
    for `confirmations` readings in a row (so a single odd window can't alarm).
    Falls back to a simple rule if no trained model is available.
    """

    def __init__(self, model: Optional[FireModel], window: int = 10, threshold: Optional[float] = None,
                 min_temp: float = 35.0, confirmations: int = 3, max_gap: float = 5.0, rule_rise: float = 5.0):
        self.model = model
        self.window = model.window if model else window
        self.threshold = threshold if threshold is not None else (model.threshold if model else 0.5)
        self.min_temp, self.confirmations, self.max_gap, self.rule_rise = min_temp, confirmations, max_gap, rule_rise
        self.buf: deque = deque(maxlen=self.window)
        self.last_ts: Optional[float] = None
        self.hits = 0

    @property
    def name(self) -> str:
        return "model" if self.model else "rules"

    def reset(self):
        self.buf.clear()
        self.hits = 0

    def _rule_score(self, w: list[float]) -> float:
        rise = w[-1] - w[0]
        steps = [b - a for a, b in zip(w, w[1:])]
        ok = (sum(1 for s in steps if s >= 0) / len(steps) >= 0.8
              and w[-1] >= max(w) - 0.3
              and all(x - w[0] >= self.rule_rise * 0.5 for x in w[-3:]))
        return 1.0 if (rise >= self.rule_rise and ok) else min(0.49, max(0.0, rise / self.rule_rise) * 0.49)

    def update(self, temp: float, now: float) -> tuple[Optional[float], bool, float]:
        """Returns (score or None while warming up, triggered, rise over the window)."""
        if self.last_ts is not None and now - self.last_ts > self.max_gap:
            self.reset()                                  # data gap: old readings no longer describe a trend
        self.last_ts = now
        self.buf.append(float(temp))
        if len(self.buf) < self.window:
            return None, False, 0.0
        w = list(self.buf)
        score = self.model.predict_proba(w) if self.model else self._rule_score(w)
        hit = score >= self.threshold and w[-1] >= self.min_temp
        self.hits = self.hits + 1 if hit else 0
        return score, self.hits >= self.confirmations, w[-1] - w[0]
