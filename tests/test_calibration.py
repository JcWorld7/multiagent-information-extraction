import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.apply_labelcentered_calibration import apply_config
from Labelcentered.calibration import select_calibrated_candidate_ids, select_label_candidate_ids
from Labelcentered.settings import LABELS


class CalibrationTests(unittest.TestCase):
    def candidate(self, cid, label, text, start=0, experts=None, confidence=0.8):
        return {
            "candidate_id": cid,
            "label": label,
            "text": text,
            "start": start,
            "end": start + len(text),
            "proposed_by_experts": experts or [],
            "confidence_by_expert": {expert: confidence for expert in experts or []},
            "threshold_by_expert": {},
            "source_section": "methods",
            "generation_origin": "expert",
        }

    def config(self, label="Outcome"):
        return {
            "label": label,
            "require_label_filter": True,
            "reject_weak_fragments": True,
            "min_vote_fraction": 0.67,
            "min_expert_support": 2,
            "min_confidence": 0.5,
            "use_high_quality_rescue": False,
            "vote_weights": {"SciBERT": 1.0, "PubMedBERT": 1.0, "ClinicalBERT": 1.0},
        }

    def test_calibrated_selector_accepts_equal_multi_expert_support(self):
        candidate = self.candidate(
            "OUT-C0001",
            "Outcome",
            "depressive symptoms",
            experts=["SciBERT", "ClinicalBERT"],
        )
        selected = select_calibrated_candidate_ids([candidate], {}, self.config())
        self.assertEqual(selected, ["OUT-C0001"])

    def test_calibrated_selector_uses_weighted_doctor_votes(self):
        candidate = self.candidate(
            "OUT-C0001",
            "Outcome",
            "anxiety symptoms",
            experts=["SciBERT"],
        )
        votes = {
            "OUT-C0001": {
                "SciBERT_Outcome_Doctor": "keep",
                "PubMedBERT_Outcome_Doctor": "keep",
                "ClinicalBERT_Outcome_Doctor": "reject",
            }
        }
        selected = select_calibrated_candidate_ids([candidate], votes, self.config())
        self.assertEqual(selected, ["OUT-C0001"])

    def test_fixed_recommended_mode_uses_cached_recommendations(self):
        candidate = self.candidate(
            "OUT-C0001",
            "Outcome",
            "anxiety symptoms",
            experts=["SciBERT"],
        )
        payload = {
            "label_candidate_registries": {label: [] for label in LABELS},
            "candidate_vote_matrices_by_label": {label: {} for label in LABELS},
            "recommended_candidate_ids_by_label": {"Outcome": ["OUT-C0001"]},
        }
        payload["label_candidate_registries"]["Outcome"] = [candidate]
        selected = select_label_candidate_ids(
            payload,
            "Outcome",
            {"label": "Outcome", "source_mode": "recommended"},
        )
        self.assertEqual(selected, ["OUT-C0001"])

    def test_apply_config_rewrites_final_spans_without_rewriting_text(self):
        text = "The outcome was depressive symptoms."
        start = text.index("depressive symptoms")
        candidate = self.candidate(
            "OUT-C0001",
            "Outcome",
            "depressive symptoms",
            start=start,
            experts=["SciBERT", "ClinicalBERT"],
        )
        registries = {label: [] for label in LABELS}
        registries["Outcome"] = [candidate]
        payload = {
            "model_input_text": text,
            "label_candidate_registries": registries,
            "candidate_vote_matrices_by_label": {label: {} for label in LABELS},
            "final_spans_by_label": {label: [] for label in LABELS},
            "final_candidate_ids_by_label": {label: [] for label in LABELS},
            "final_spans": [],
            "unresolved_candidates": [],
            "overall_consensus_reached": False,
        }
        config = {
            "schema_version": 1,
            "policy_name": "calibrated_lightweight_candidate_selector",
            "selected_variant": "default",
            "labels": {label: self.config(label) for label in LABELS},
        }
        output = apply_config(payload, config)
        self.assertEqual(output["final_selection_policy"], "calibrated")
        self.assertEqual(output["final_candidate_ids_by_label"]["Outcome"], ["OUT-C0001"])
        self.assertEqual(output["final_spans_by_label"]["Outcome"][0]["text"], "depressive symptoms")


if __name__ == "__main__":
    unittest.main()
