import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.output_validation import validate_output
from Labelcentered.settings import LABELS


class OutputValidationTests(unittest.TestCase):
    def base_payload(self):
        text = "Outcome score was measured."
        span = {"candidate_id": "OUT-C0001", "label": "Outcome", "text": "Outcome score", "start": 0, "end": 13}
        registries = {label: [] for label in LABELS}
        registries["Outcome"] = [{**span, "confidence_by_expert": {}, "threshold_by_expert": {}}]
        by_label = {label: [] for label in LABELS}
        by_label["Outcome"] = [span]
        ids = {label: [] for label in LABELS}
        ids["Outcome"] = ["OUT-C0001"]
        return {
            "model_input_text": text,
            "label_candidate_registries": registries,
            "final_spans_by_label": by_label,
            "final_candidate_ids_by_label": ids,
            "final_spans": [span],
            "unresolved_candidates": [],
            "overall_consensus_reached": False,
        }

    def test_every_final_span_exact(self):
        self.assertEqual(validate_output(self.base_payload()), [])

    def test_invalid_text_offset_detected(self):
        payload = self.base_payload()
        payload["final_spans"][0]["text"] = "Wrong"
        payload["final_spans_by_label"]["Outcome"][0]["text"] = "Wrong"
        self.assertTrue(validate_output(payload))

    def test_summary_separation(self):
        payload = self.base_payload()
        payload["generated_summaries"] = {
            "overall_summary": "Outcome was measured.",
            "summary_based_on_final_spans": True,
            "summary_final_selection_policy": "calibrated",
        }
        self.assertEqual(validate_output(payload), [])

    def test_empty_label_cannot_be_marked_consensus(self):
        payload = self.base_payload()
        payload["label_consensus_status"] = {label: False for label in LABELS}
        payload["label_consensus_status"]["StatisticalAnalysis"] = True
        errors = validate_output(payload)
        self.assertTrue(any("StatisticalAnalysis cannot have consensus=true" in error for error in errors))

    def test_final_candidate_requires_three_keep_votes(self):
        payload = self.base_payload()
        payload["candidate_vote_matrices_by_label"] = {
            label: {} for label in LABELS
        }
        payload["candidate_vote_matrices_by_label"]["Outcome"] = {
            "OUT-C0001": {
                "A": "keep",
                "B": "keep",
                "C": "reject",
            }
        }
        errors = validate_output(payload)
        self.assertTrue(any("not supported by three keep votes" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
