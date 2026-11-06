import os, sys, tempfile, unittest
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

if __name__ == "__main__":
    unittest.main()
