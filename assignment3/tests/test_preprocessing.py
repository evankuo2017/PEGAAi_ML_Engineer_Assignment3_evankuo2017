import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
from preprocessing import load_data, clean_data, cleaning_summary_to_text, REQUIRED_COLUMNS, SENSOR_COLUMNS


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.csv = os.path.join(ROOT, "data", "sensor_data.csv")

    def test_load_data_schema(self):
        df = load_data(self.csv)
        for c in REQUIRED_COLUMNS:
            self.assertIn(c, df.columns)
        self.assertEqual(len(df), 300)

    def test_clean_handles_missing(self):
        df = load_data(self.csv)
        cleaned, summary = clean_data(df)
        for c in SENSOR_COLUMNS:
            self.assertEqual(int(cleaned[c].isna().sum()), 0)
            self.assertGreaterEqual(summary.missing_before[c], 0)
            self.assertEqual(summary.missing_after[c], 0)
        self.assertIn("_z", "".join(cleaned.columns))  # z columns exist
        for c in SENSOR_COLUMNS:
            self.assertIn(f"{c}_z", cleaned.columns)

    def test_clean_sorted_timestamps(self):
        df = load_data(self.csv)
        cleaned, _ = clean_data(df)
        ts = list(cleaned["timestamp"])
        self.assertEqual(ts, sorted(ts))

    def test_summary_text(self):
        df = load_data(self.csv)
        _, summary = clean_data(df)
        txt = cleaning_summary_to_text(summary)
        self.assertIn("Rows:", txt)
        self.assertIn("Missing values before cleaning:", txt)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_data(os.path.join(ROOT, "data", "nope.csv"))

    def test_clean_empty(self):
        with self.assertRaises(ValueError):
            clean_data(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
