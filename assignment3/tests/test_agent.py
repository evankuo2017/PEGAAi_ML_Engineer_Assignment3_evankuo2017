import os, sys, tempfile, unittest
from unittest import mock
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agent import AlertAgent

class TestAgent(unittest.TestCase):
    def test_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as out_dir:
            agent = AlertAgent(os.path.join(ROOT, "data", "sensor_data.csv"), out_dir)
            stats, alerts = agent.run()
            self.assertEqual(stats["total"], 300)
            self.assertGreater(stats["anomalies"], 0)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "alerts.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "run_log.txt")))
    def test_stream_mode_emits_each_alert(self):
        with tempfile.TemporaryDirectory() as out_dir, mock.patch("agent.time.sleep") as sleep:
            agent = AlertAgent(os.path.join(ROOT, "data", "sensor_data.csv"), out_dir, stream=True)
            stats, alerts = agent.run()
            self.assertEqual(sleep.call_count, len(alerts))  # one pause per streamed alert

if __name__ == "__main__":
    unittest.main()
