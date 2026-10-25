"""
Train the sudden-temperature-spike (fire) detector.

  python train_fire_model.py                      # synthetic bootstrap data only
  python train_fire_model.py --readings readings.csv --events events.csv   # + your real recordings

Needs (training only, NOT on the server):  pip install numpy scikit-learn

Outputs (next to this script by default):
  fire_model.json      model the server loads at startup
  fire_dataset.csv     the labelled windows used (t0..t9 = readings, label 1 = spike/fire)
  training_report.md   dataset description, model, metrics (use in your project report)

REAL DATA (recommended for your report: the course asks for a dataset you collected)
  1. On the server set SENSOR_LOG_INTERVAL=1 so every 1 s reading is saved, and record for a while:
     normal conditions, plus staged heat events (hair dryer from a distance, hot water cup near the sensor).
  2. Export readings:
       \\copy (select created_at, temperature from sensor_logs where device_id='living-room'
               and temperature is not null order by created_at) to 'readings.csv' csv header
  3. Write events.csv listing when you applied heat (only the period where temperature was rising):
       start,end
       2026-10-21 15:02:10,2026-10-21 15:02:40
  4. Run this script with --readings readings.csv --events events.csv
"""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fire_model import FEATURES, SpikeDetector, extract_features

W = 10          # window length (readings)
L = W + 10      # raw sequence length before taking the last W (lets the sensor-lag filter settle)


# ----------------------------------------------------------------------------
# Synthetic bootstrap data
# ----------------------------------------------------------------------------
def _finish(raw: np.ndarray, rng, noise: float) -> list[float]:
    """Apply sensor response lag, measurement noise and resolution (DHT22 0.1 C / DHT11 1 C); keep last W."""
    a = rng.uniform(0.55, 1.0)                              # 1.0 = instant response
    y = raw.copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + a * (raw[i] - y[i - 1])
    y = y + rng.normal(0, noise, len(y))
    step = 1.0 if rng.random() < 0.15 else 0.1
    return list(np.round(y / step) * step)[-W:]


def make_normal(rng) -> list[float]:
    base = rng.uniform(20, 38)
    t = np.arange(L, dtype=float)
    noise = rng.uniform(0.05, 0.35)
    kind = rng.choice(["steady", "drift", "warming", "cooling", "glitch", "oscillation", "hot_steady", "small_step"],
                      p=[0.26, 0.18, 0.12, 0.08, 0.14, 0.08, 0.07, 0.07])
    raw = np.full(L, base)
    if kind == "drift":
        raw = base + rng.uniform(-0.15, 0.15) * t
    elif kind == "warming":                                  # hand, cup of tea, heater far away
        raw = base + rng.uniform(1.0, 3.5) / (W - 1) * np.maximum(t - (L - W), 0)
    elif kind == "cooling":
        raw = base + rng.uniform(-1.0, -0.2) * np.maximum(t - (L - W), 0)
    elif kind == "oscillation":                              # AC cycling, draught
        raw = base + rng.uniform(0.3, 1.5) * np.sin(2 * math.pi * t / rng.uniform(6, 30) + rng.uniform(0, 6))
    elif kind == "hot_steady":
        raw = np.full(L, rng.uniform(36, 45))
    elif kind == "small_step":                               # door opened, fan switched on
        raw = base + rng.choice([-1, 1]) * rng.uniform(1.5, 3.5) * (t >= L - rng.integers(2, W))
    out = _finish(raw, rng, noise)
    if kind == "glitch":                                     # one or two wild readings
        # often at the very end of the window: the hardest case, it looks like the start of a spike
        pool = [W - 1, W - 2, W - 3] if rng.random() < 0.6 else list(range(W))
        for i in rng.choice(pool, size=rng.integers(1, 3), replace=False):
            out[i] += rng.choice([-1, 1]) * rng.uniform(3, 15)
    return out


def make_fire(rng) -> list[float]:
    for _ in range(50):
        base = rng.uniform(20, 38)
        t = np.arange(L, dtype=float) - (L - W + rng.integers(0, 5))     # onset within first 5 samples of window
        tp = np.maximum(t, 0)
        noise = rng.uniform(0.05, 0.35)
        kind = rng.choice(["ramp", "step", "exp", "accel", "noisy_slow"], p=[0.35, 0.20, 0.20, 0.15, 0.10])
        if kind == "ramp":
            raw = base + rng.uniform(0.5, 3.0) * tp
        elif kind == "step":
            raw = base + rng.uniform(4, 20) * (t >= 0) + rng.uniform(0, 0.5) * tp
        elif kind == "exp":
            raw = base + rng.uniform(8, 40) * (1 - np.exp(-tp / rng.uniform(2, 8)))
        elif kind == "accel":
            raw = base + rng.uniform(0.05, 0.35) * tp ** 2
        else:
            raw = base + rng.uniform(0.5, 1.0) * tp
            noise = rng.uniform(0.3, 0.6)
        out = _finish(raw, rng, noise)
        if out[-1] - out[0] >= 4.0:
            return out
    return out


def synthetic_windows(n: int, rng) -> tuple[list[list[float]], list[int]]:
    n_fire = n // 3
    wins = [make_fire(rng) for _ in range(n_fire)] + [make_normal(rng) for _ in range(n - n_fire)]
    return wins, [1] * n_fire + [0] * (n - n_fire)


# ----------------------------------------------------------------------------
# Real recordings
# ----------------------------------------------------------------------------
def _ts(s: str) -> float:
    s = s.strip().replace("Z", "+00:00")
    if len(s) > 3 and s[-3] == "+":                          # postgres '+00' -> '+00:00'
        s += ":00"
    dt = datetime.fromisoformat(s)
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def real_windows(readings_csv: str, events_csv: str):
    with open(readings_csv, newline="") as f:
        rows = [(_ts(r[0]), float(r[1])) for r in list(csv.reader(f))[1:] if len(r) >= 2 and r[1] != ""]
    rows.sort()
    with open(events_csv, newline="") as f:
        events = [(_ts(r[0]), _ts(r[1])) for r in list(csv.reader(f))[1:] if len(r) >= 2]

    segments, cur = [], [rows[0]] if rows else []
    for prev, row in zip(rows, rows[1:]):                    # split where readings are not ~1 s apart
        if row[0] - prev[0] > 5:
            segments.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segments.append(cur)

    wins, labels = [], []
    for seg in segments:
        for i in range(W - 1, len(seg)):
            w = [v for _, v in seg[i - W + 1: i + 1]]
            t_last = seg[i][0]
            wins.append(w)
            labels.append(1 if any(a <= t_last <= b for a, b in events) else 0)
    return wins, labels


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def metrics(y, pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4) if proba is not None else None,
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def pick_threshold(y, proba) -> float:
    """Lowest threshold that keeps precision >= 0.98 on validation data (false alarms are costly); else best F1."""
    grid = np.arange(0.30, 0.961, 0.01)
    best_f1, best_t = -1.0, 0.5
    for t in grid:
        pred = proba >= t
        if precision_score(y, pred, zero_division=0) >= 0.98 and pred.any():
            return float(round(t, 2))
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(round(t, 2))
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=9000, help="number of synthetic windows (0 = none)")
    ap.add_argument("--readings", help="CSV: timestamp,temperature (1 reading per second)")
    ap.add_argument("--events", help="CSV: start,end of staged heat events")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)

    windows, labels, sources = [], [], []
    if args.synthetic > 0:
        w, l = synthetic_windows(args.synthetic, rng)
        windows += w; labels += l; sources += ["synthetic"] * len(w)
    if args.readings:
        if not args.events:
            ap.error("--readings needs --events")
        w, l = real_windows(args.readings, args.events)
        windows += w; labels += l; sources += ["recorded"] * len(w)
    if not windows:
        ap.error("no data: use --synthetic and/or --readings")

    X = np.array([extract_features(w) for w in windows])
    y = np.array(labels)
    idx = np.arange(len(y))
    i_train, i_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=args.seed)
    i_val, i_test = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=args.seed)

    scaler = StandardScaler().fit(X[i_train])
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(scaler.transform(X[i_train]), y[i_train])
    p_val = lr.predict_proba(scaler.transform(X[i_val]))[:, 1]
    thr = pick_threshold(y[i_val], p_val)
    p_test = lr.predict_proba(scaler.transform(X[i_test]))[:, 1]
    lr_val, lr_test = metrics(y[i_val], p_val >= thr, p_val), metrics(y[i_test], p_test >= thr, p_test)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=args.seed)
    rf.fit(X[i_train], y[i_train])
    p_rf = rf.predict_proba(X[i_test])[:, 1]
    rule = SpikeDetector(None, window=W, min_temp=-1e9)       # simple threshold rule as a baseline
    p_rule = np.array([rule._rule_score(windows[i]) for i in i_test])
    comparison = {
        "random_forest": metrics(y[i_test], p_rf >= 0.5, p_rf),
        "rule_baseline": metrics(y[i_test], p_rule >= 0.5, p_rule),
    }

    n_real = sources.count("recorded")
    dataset = {
        "total_windows": int(len(y)), "spike": int(y.sum()), "normal": int(len(y) - y.sum()),
        "window_readings": W, "sampling": "1 reading per second",
        "synthetic_windows": sources.count("synthetic"), "recorded_windows": n_real,
        "split": {"train": int(len(i_train)), "validation": int(len(i_val)), "test": int(len(i_test))},
    }
    spec = {
        "model_type": "logistic_regression", "window": W, "features": FEATURES, "threshold": thr,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "metrics": {"validation": lr_val, "test": lr_test}, "comparison": comparison,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "fire_model.json").write_text(json.dumps(spec, indent=2))

    with open(out / "fire_dataset.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow([f"t{i}" for i in range(W)] + ["label", "source"])
        for w, lab, src in zip(windows, labels, sources):
            cw.writerow([f"{v:.1f}" for v in w] + [lab, src])

    top = sorted(zip(FEATURES, lr.coef_[0]), key=lambda kv: -abs(kv[1]))[:5]
    report = f"""# Fire / sudden temperature spike detector: training report

## Dataset
- Windows: {dataset['total_windows']} ({dataset['spike']} spike, {dataset['normal']} normal), each = the last {W} temperature readings at 1 reading/second
- Sources: {dataset['synthetic_windows']} synthetic windows, {dataset['recorded_windows']} windows from real sensor recordings
- Split: train {dataset['split']['train']}, validation {dataset['split']['validation']}, test {dataset['split']['test']} (stratified)
- Synthetic data simulates DHT22/DHT11 sensors (noise, 0.1 C / 1 C resolution, response lag) under normal conditions
  (steady, drift, hand/cup warming, cooling, door/AC changes, single-reading glitches, hot steady rooms)
  and spike conditions (linear ramps, step jumps, exponential and accelerating rises, noisy slow rises).
{"- **No real recordings were included in this run.** Metrics below only describe the synthetic data." if n_real == 0 else "- Real recordings were included in training and testing."}

## Model
- Logistic regression on {len(FEATURES)} shape features of the window ({', '.join(FEATURES)}), standardised.
- Purpose: decide, for every new sensor reading, whether the last {W} readings show a sudden, sustained temperature rise (possible fire).
- Decision threshold {thr} (lowest score keeping validation precision >= 0.98). Strongest features: {', '.join(f'{n} ({c:+.2f})' for n, c in top)}.
- At run time the server also requires temperature >= a safety minimum and 3 consecutive positive windows.

## Results on held-out test set (threshold {thr})
| Model | Accuracy | Precision | Recall | F1 | ROC AUC | False-positive rate |
|---|---|---|---|---|---|---|
| Logistic regression (deployed) | {lr_test['accuracy']} | {lr_test['precision']} | {lr_test['recall']} | {lr_test['f1']} | {lr_test['roc_auc']} | {lr_test['false_positive_rate']} |
| Random forest (comparison) | {comparison['random_forest']['accuracy']} | {comparison['random_forest']['precision']} | {comparison['random_forest']['recall']} | {comparison['random_forest']['f1']} | {comparison['random_forest']['roc_auc']} | {comparison['random_forest']['false_positive_rate']} |
| Simple threshold rule (baseline) | {comparison['rule_baseline']['accuracy']} | {comparison['rule_baseline']['precision']} | {comparison['rule_baseline']['recall']} | {comparison['rule_baseline']['f1']} | {comparison['rule_baseline']['roc_auc']} | {comparison['rule_baseline']['false_positive_rate']} |

Confusion matrix (deployed model): {lr_test['confusion_matrix']}

## Limitations
- A temperature sensor only reacts to heat near it, with a delay. This is a demonstration, not a certified fire alarm.
- Performance on real rooms depends on real recordings: retrain with `--readings/--events` using your own data.
"""
    (out / "training_report.md").write_text(report)

    print(f"windows={len(y)} spike={int(y.sum())} real={n_real} threshold={thr}")
    print("test:", json.dumps({k: v for k, v in lr_test.items() if k != 'confusion_matrix'}), lr_test["confusion_matrix"])
    print("random forest:", comparison["random_forest"]["f1"], "| rule baseline f1:", comparison["rule_baseline"]["f1"])
    print("wrote fire_model.json, fire_dataset.csv, training_report.md to", out)


if __name__ == "__main__":
    main()
