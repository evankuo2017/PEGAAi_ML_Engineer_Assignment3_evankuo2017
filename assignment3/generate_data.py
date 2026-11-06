# -*- coding: utf-8 -*-
"""Generate reproducible simulated sensor data for a smart factory.

Requires Python >= 3.9.
"""

import argparse
import os

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate smart-factory sensor CSV data.")
    parser.add_argument("--rows", type=int, default=300, help="Number of rows to generate (default: 300).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output", type=str, default="data/sensor_data.csv", help="Output CSV path (default: data/sensor_data.csv).")
    parser.add_argument("--missing-rate", type=float, default=0.05, help="Missing value rate per sensor column (default: 0.05).")
    parser.add_argument("--anomaly-rate", type=float, default=0.10, help="Anomaly row rate (default: 0.10).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    n_rows = args.rows
    start_ts = pd.Timestamp("2024-06-03 19:05:00")
    timestamps = [(start_ts + pd.Timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(n_rows)]

    n_anomaly = int(round(n_rows * args.anomaly_rate))
    if n_anomaly < 1 and n_rows > 0:
        n_anomaly = 1
    if n_anomaly > n_rows:
        n_anomaly = n_rows
    anomaly_indices = set(int(x) for x in rng.choice(n_rows, size=n_anomaly, replace=False))

    temp_vals = np.empty(n_rows, dtype=float)
    pressure_vals = np.empty(n_rows, dtype=float)
    vibration_vals = np.empty(n_rows, dtype=float)
    labels = ["normal"] * n_rows

    for i in range(n_rows):
        is_abnormal = i in anomaly_indices
        if not is_abnormal:
            labels[i] = "normal"
            temp_vals[i] = rng.uniform(45.0, 50.0)
            pressure_vals[i] = rng.uniform(1.00, 1.05)
            vibration_vals[i] = rng.uniform(0.02, 0.04)
        else:
            labels[i] = "abnormal"
            abnormal_mask = rng.random(3) < 0.5
            if not abnormal_mask.any():
                abnormal_mask[int(rng.integers(0, 3))] = True

            if abnormal_mask[0]:
                if rng.random() < 0.5:
                    temp_vals[i] = rng.uniform(53.0, 58.0)
                else:
                    temp_vals[i] = rng.uniform(38.0, 42.0)
            else:
                temp_vals[i] = rng.uniform(45.0, 50.0)

            if abnormal_mask[1]:
                if rng.random() < 0.5:
                    pressure_vals[i] = rng.uniform(1.09, 1.18)
                else:
                    pressure_vals[i] = rng.uniform(0.90, 0.96)
            else:
                pressure_vals[i] = rng.uniform(1.00, 1.05)

            if abnormal_mask[2]:
                vibration_vals[i] = rng.uniform(0.08, 0.15)
            else:
                vibration_vals[i] = rng.uniform(0.02, 0.04)

    temp_vals = np.round(temp_vals, 1)
    pressure_vals = np.round(pressure_vals, 2)
    vibration_vals = np.round(vibration_vals, 2)

    n_missing_per_col = int(round(n_rows * args.missing_rate))
    if n_missing_per_col < 1 and n_rows > 0 and args.missing_rate > 0:
        n_missing_per_col = 1

    def choose_rows() -> set:
        size = min(n_missing_per_col, n_rows)
        if size <= 0:
            return set()
        return set(int(x) for x in rng.choice(n_rows, size=size, replace=False))

    missing_temp_rows = choose_rows()
    missing_pressure_rows = choose_rows()
    missing_vibration_rows = choose_rows()

    temp_str = []
    pressure_str = []
    vibration_str = []
    for i in range(n_rows):
        temp_str.append("" if i in missing_temp_rows else f"{temp_vals[i]:.1f}")
        pressure_str.append("" if i in missing_pressure_rows else f"{pressure_vals[i]:.2f}")
        vibration_str.append("" if i in missing_vibration_rows else f"{vibration_vals[i]:.2f}")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp": temp_str,
        "pressure": pressure_str,
        "vibration": vibration_str,
        "label": labels,
    })

    out_path = args.output
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    missing_temp = sum(1 for v in temp_str if v == "")
    missing_pressure = sum(1 for v in pressure_str if v == "")
    missing_vibration = sum(1 for v in vibration_str if v == "")
    normal_count = labels.count("normal")
    abnormal_count = labels.count("abnormal")

    print(f"Total rows: {len(df)}")
    print(f"normal: {normal_count}, abnormal: {abnormal_count}")
    print(f"Missing counts -> temp: {missing_temp}, pressure: {missing_pressure}, vibration: {missing_vibration}")
    print("Preview (first 5 rows):")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
