"""Smart Factory Alert Agent - CLI orchestrator."""
from __future__ import annotations
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

try:
    from .preprocessing import load_data, clean_data, cleaning_summary_to_text
    from .detector import detect, anomalies_to_dataframe, summarize, compute_metrics
except ImportError:
    from preprocessing import load_data, clean_data, cleaning_summary_to_text
    from detector import detect, anomalies_to_dataframe, summarize, compute_metrics


STREAM_DELAY_SEC = 0.3  # delay between alerts in --stream mode

# Simple tool registry to make the agent's perception/reasoning/action explicit.
TOOLS: Dict[str, Callable] = {}


def tool(name):
    def decorator(fn):
        TOOLS[name] = fn
        return fn
    return decorator


@tool("load_data")
def load_data_tool(path):
    return load_data(path)


@tool("clean_data")
def clean_data_tool(df):
    return clean_data(df)


@tool("detect")
def detect_tool(df):
    return detect(df)


@tool("save_results")
def save_results_tool(alerts_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "alerts.csv"
    alerts_df.to_csv(csv_path, index=False)
    return csv_path


class AlertAgent:
    def __init__(self, input_path, output_dir="outputs", log_path=None, stream=False):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.log_path = Path(log_path) if log_path else self.output_dir / "run_log.txt"
        self.log_lines = []
        self.stream = stream

    def _log(self, msg):
        self.log_lines.append(msg)
        print(msg)

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Smart Factory Alert Agent started")
        self._log(f"Input: {self.input_path}")

        df = TOOLS["load_data"](str(self.input_path))
        self._log(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

        cleaned, csummary = TOOLS["clean_data"](df)
        self._log(cleaning_summary_to_text(csummary))

        results = TOOLS["detect"](cleaned)
        stats = summarize(results)
        self._log("\n=== Detection Summary ===")
        self._log(f"Total records: {stats['total']}")
        self._log(f"Anomalies: {stats['anomalies']} ({stats['anomaly_rate']}%)")
        self._log(f"By severity: {stats['by_severity']}")
        self._log(f"By sensor: {stats['by_sensor']}")
        self._log(f"Avg score: {stats['avg_score']}  Max score: {stats['max_score']}")

        metrics = compute_metrics(results)
        self._log("\n=== Evaluation Metrics (vs. label) ===")
        self._log(f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
        self._log(f"Precision={metrics['precision']}  Recall={metrics['recall']}  F1={metrics['f1']}")

        alerts_df = anomalies_to_dataframe(results)
        self._log("\n=== Anomaly Alerts ===")
        if alerts_df.empty:
            self._log("No anomalies detected.")
        else:
            for _, r in alerts_df.iterrows():
                self._log(
                    f"{r['timestamp']} | score={r['score']} | severity={r['severity']} "
                    f"| sensors={r['abnormal_sensors']} | temp={r['temp']} pressure={r['pressure']} "
                    f"vibration={r['vibration']}"
                )
                self._log(f"   -> {r['suggestions']}")
                if self.stream:
                    time.sleep(STREAM_DELAY_SEC)  # stream mode: emit alerts one by one
            csv_path = TOOLS["save_results"](alerts_df, str(self.output_dir))
            self._log(f"\nAlerts saved to: {csv_path}")

        self.log_path.write_text("\n".join(self.log_lines), encoding="utf-8")
        self._log(f"Run log saved to: {self.log_path}")
        return stats, alerts_df


def build_arg_parser():
    p = argparse.ArgumentParser(description="Smart Factory Alert Agent")
    p.add_argument("--input", "-i", default="data/sensor_data.csv", help="Path to sensor CSV")
    p.add_argument("--output", "-o", default="outputs", help="Output directory")
    p.add_argument("--log", default=None, help="Path to run log file")
    p.add_argument("--stream", action="store_true", default=False, help="Stream sensor results one by one")
    return p


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    args = build_arg_parser().parse_args()
    # If launched from another working directory (e.g. `python assignment3/main.py`),
    # resolve relative paths against the project root so the defaults still work.
    if not Path(args.input).exists() and (PROJECT_ROOT / args.input).exists():
        args.input = str(PROJECT_ROOT / args.input)
        args.output = str(PROJECT_ROOT / args.output)
        if args.log:
            args.log = str(PROJECT_ROOT / args.log)
    agent = AlertAgent(args.input, args.output, args.log, args.stream)
    agent.run()


if __name__ == "__main__":
    main()
