import os, sys, unittest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pandas as pd
from detector import detect, summarize, anomalies_to_dataframe

class TestDetector(unittest.TestCase):
    def _df(self, rows):
        return pd.DataFrame(rows)
    def test_normal(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":48.0,"pressure":1.02,"vibration":0.03,"label":"normal"}])
        r = detect(df)[0]
        self.assertFalse(r.is_anomaly); self.assertEqual(r.score,0.0); self.assertEqual(r.severity,"none")
    def test_temp_high(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":55.0,"pressure":1.02,"vibration":0.03,"label":"abnormal"}])
        r = detect(df)[0]
        self.assertTrue(r.is_anomaly); self.assertIn("temp", r.abnormal_sensors)
    def test_temp_low(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":40.0,"pressure":1.02,"vibration":0.03,"label":"abnormal"}])
        self.assertTrue(detect(df)[0].is_anomaly)
    def test_pressure(self):
        for v in [1.10,0.95]:
            df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":48.0,"pressure":v,"vibration":0.03,"label":"abnormal"}])
            self.assertTrue(detect(df)[0].is_anomaly, msg=str(v))
    def test_vib(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":48.0,"pressure":1.02,"vibration":0.09,"label":"abnormal"}])
        self.assertTrue(detect(df)[0].is_anomaly)
    def test_boundary(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":52.0,"pressure":1.08,"vibration":0.07,"label":"normal"}])
        self.assertFalse(detect(df)[0].is_anomaly)
    def test_multi(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":55.0,"pressure":1.10,"vibration":0.09,"label":"abnormal"}])
        r = detect(df)[0]
        self.assertEqual(r.severity,"high"); self.assertEqual(len(r.abnormal_sensors),3)
    def test_score_scale_consistent(self):
        # temp +5 over threshold and vibration +0.02 over threshold are both half a
        # deviation scale, so they must receive the same score.
        t = detect(self._df([{"timestamp":"2024-06-03 19:05:00","temp":57.0,"pressure":1.02,"vibration":0.03,"label":"abnormal"}]))[0]
        v = detect(self._df([{"timestamp":"2024-06-03 19:05:00","temp":48.0,"pressure":1.02,"vibration":0.09,"label":"abnormal"}]))[0]
        self.assertAlmostEqual(t.score, v.score, places=2)
        self.assertEqual(t.score, 70.0); self.assertEqual(t.severity, "medium")
    def test_missing_col(self):
        df = self._df([{"timestamp":"2024-06-03 19:05:00","temp":48.0}])
        with self.assertRaises(ValueError):
            detect(df)
    def test_adf(self):
        df = self._df([
            {"timestamp":"2024-06-03 19:05:00","temp":48.0,"pressure":1.02,"vibration":0.03,"label":"normal"},
            {"timestamp":"2024-06-03 19:10:00","temp":55.0,"pressure":1.02,"vibration":0.03,"label":"abnormal"},
        ])
        adf = anomalies_to_dataframe(detect(df))
        self.assertEqual(len(adf),1)
    def test_summary(self):
        from preprocessing import load_data, clean_data
        cleaned,_ = clean_data(load_data(os.path.join(ROOT,"data","sensor_data.csv")))
        s = summarize(detect(cleaned))
        self.assertEqual(s["total"],300); self.assertGreater(s["anomalies"],0)

if __name__=="__main__":
    unittest.main()
