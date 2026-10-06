import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.consistency import consistency_metrics
from Labelcentered.global_meta_agent import detect_cross_label_conflicts


class ConsistencyAndConflictTests(unittest.TestCase):
    def test_missing_candidate_not_treated_as_rejection(self):
        reviews = [
            {"agent_role": "A", "decisions_by_candidate": {"C1": "keep"}},
            {"agent_role": "B", "decisions_by_candidate": {"C1": "keep"}},
            {"agent_role": "C", "decisions_by_candidate": {}},
        ]
        metrics = consistency_metrics(reviews, ["C1"])
        self.assertEqual(metrics["missing_decision_candidate_ids"], ["C1"])
        self.assertEqual(metrics["conflicted_candidate_ids"], [])

    def test_cross_label_conflict_detection(self):
        span = {"candidate_id": "OUT-C0001", "text": "stress score", "start": 10, "end": 22}
        other = {"candidate_id": "STA-C0001", "text": "stress score", "start": 10, "end": 22}
        conflicts = detect_cross_label_conflicts({"Outcome": [span], "StatisticalAnalysis": [other]})
        self.assertEqual(conflicts[0]["type"], "exact_duplicate_cross_label")


if __name__ == "__main__":
    unittest.main()

