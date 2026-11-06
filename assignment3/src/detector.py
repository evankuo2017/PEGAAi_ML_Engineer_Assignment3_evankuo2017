"""Rule-based anomaly detection for smart factory sensor data."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

SENSOR_COLUMNS = ["temp", "pressure", "vibration"]
ABNORMAL_THRESHOLDS = {"temp": (43.0, 52.0), "pressure": (0.97, 1.08), "vibration": (None, 0.07)}
SEVERITY_LEVELS = ("low", "medium", "high")
SUGGESTIONS = {
    "temp": {"high": "Temperature critically high: inspect cooling system.", "low": "Temperature too low: check heater/set-point."},
    "pressure": {"high": "Pressure too high: inspect relief valve.", "low": "Pressure too low: check pump/leaks."},
    "vibration": {"high": "Excessive vibration: inspect bearings/alignment."},
}


@dataclass
class AnomalyResult:
    timestamp: pd.Timestamp
    temp: float
    pressure: float
    vibration: float
    label: str
    is_anomaly: bool
    score: float
    severity: str
    abnormal_sensors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


def _abnormal(sensor, value):
    lo, hi = ABNORMAL_THRESHOLDS[sensor]
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False


# Deviation scale per sensor = 2 x width of the normal operating range
# (temp 45-50 -> 10, pressure 1.00-1.05 -> 0.10, vibration 0.02-0.04 -> 0.04).
# Normalising by range width (instead of the threshold value) puts all sensors on
# a comparable scale, e.g. temp +5°C and vibration +0.02 over threshold both give 0.5.
DEVIATION_SCALE = {"temp": 10.0, "pressure": 0.10, "vibration": 0.04}


def _deviation(sensor, value):
    lo, hi = ABNORMAL_THRESHOLDS[sensor]
    scale = DEVIATION_SCALE[sensor]
    if lo is not None and value < lo:
        return min(1.0, (lo - value) / scale)
    if hi is not None and value > hi:
        return min(1.0, (value - hi) / scale)
    return 0.0


def _direction(sensor, value):
    lo, hi = ABNORMAL_THRESHOLDS[sensor]
    if hi is not None and value > hi:
        return "high"
    if lo is not None and value < lo:
        return "low"
    return "high"
def detect(df):
    required = ["timestamp", "temp", "pressure", "vibration", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []
    for _, row in df.iterrows():
        abnormal = []
        deviations = []
        for sensor in SENSOR_COLUMNS:
            val = float(row[sensor])
            if _abnormal(sensor, val):
                abnormal.append(sensor)
            deviations.append(_deviation(sensor, val))

        is_anom = len(abnormal) > 0
        base = 40.0 * len(abnormal)
        dev_bonus = 60.0 * (max(deviations) if deviations else 0.0)
        score = min(100.0, base + dev_bonus) if is_anom else 0.0

        if not is_anom:
            severity = "none"
        elif len(abnormal) >= 2 or score >= 80:
            severity = "high"
        elif score >= 50:
            severity = "medium"
        else:
            severity = "low"

        suggestions = []
        for sensor in abnormal:
            d = _direction(sensor, float(row[sensor]))
            sug = SUGGESTIONS.get(sensor, {}).get(d)
            if sug and sug not in suggestions:
                suggestions.append(sug)

        results.append(AnomalyResult(
            timestamp=pd.Timestamp(row["timestamp"]),
            temp=float(row["temp"]),
            pressure=float(row["pressure"]),
            vibration=float(row["vibration"]),
            label=str(row["label"]),
            is_anomaly=is_anom,
            score=round(score, 2),
            severity=severity,
            abnormal_sensors=abnormal,
            suggestions=suggestions,
        ))
    return results


def anomalies_to_dataframe(results):
    rows = []
    for r in results:
        if not r.is_anomaly:
            continue
        rows.append({
            "timestamp": r.timestamp,
            "temp": r.temp,
            "pressure": r.pressure,
            "vibration": r.vibration,
            "label": r.label,
            "score": r.score,
            "severity": r.severity,
            "abnormal_sensors": ",".join(r.abnormal_sensors),
            "suggestions": " | ".join(r.suggestions),
        })
    return pd.DataFrame(rows)


def summarize(results):
    total = len(results)
    anom = [r for r in results if r.is_anomaly]
    by_sev = {s: 0 for s in SEVERITY_LEVELS}
    for r in anom:
        if r.severity in by_sev:
            by_sev[r.severity] += 1
    sensor_counts = {s: 0 for s in SENSOR_COLUMNS}
    for r in anom:
        for s in r.abnormal_sensors:
            sensor_counts[s] += 1
    return {
        "total": total,
        "anomalies": len(anom),
        "anomaly_rate": round(len(anom) / total * 100, 2) if total else 0.0,
        "by_severity": by_sev,
        "by_sensor": sensor_counts,
        "avg_score": round(sum(r.score for r in anom) / len(anom), 2) if anom else 0.0,
        "max_score": round(max((r.score for r in anom), default=0.0), 2),
    }


def compute_metrics(results):
    """Compute precision/recall/F1 treating 'abnormal' label as ground truth."""
    tp = fp = fn = tn = 0
    for r in results:
        gt = str(r.label).lower() == "abnormal"
        pred = bool(r.is_anomaly)
        if gt and pred:
            tp += 1
        elif (not gt) and pred:
            fp += 1
        elif gt and (not pred):
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


__all__ = ["SENSOR_COLUMNS", "ABNORMAL_THRESHOLDS", "DEVIATION_SCALE", "SEVERITY_LEVELS", "AnomalyResult",
           "detect", "anomalies_to_dataframe", "summarize", "compute_metrics"]

