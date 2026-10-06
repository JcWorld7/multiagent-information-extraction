import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.selection_policy import passes_label_filter, select_expert_consensus_verified, select_pubmedbert_plus_verified


class SelectionPolicyTests(unittest.TestCase):
    def candidate(self, cid, label, text, start=0, experts=None):
        return {
            "candidate_id": cid,
            "label": label,
            "text": text,
            "start": start,
            "end": start + len(text),
            "proposed_by_experts": experts or [],
        }

    def test_pubmedbert_candidates_are_kept_only_if_label_filter_passes(self):
        good = self.candidate("SAM-C0001", "SampleSize", "120 participants", experts=["PubMedBERT"])
        bad = self.candidate("SAM-C0002", "SampleSize", "p = .05", start=50, experts=["PubMedBERT"])
        ids = select_pubmedbert_plus_verified([good, bad], {})
        self.assertEqual(ids, ["SAM-C0001"])

    def test_non_pubmedbert_candidate_requires_majority_and_filter(self):
        pubmed = self.candidate("OUT-C0001", "Outcome", "depressive symptoms", experts=["PubMedBERT"])
        rescue = self.candidate("OUT-C0002", "Outcome", "anxiety symptoms", start=40, experts=["SciBERT"])
        weak = self.candidate("OUT-C0003", "Outcome", "linear regression", start=80, experts=["ClinicalBERT"])
        matrix = {
            "OUT-C0002": {"A": "keep", "B": "keep", "C": "reject"},
            "OUT-C0003": {"A": "keep", "B": "keep", "C": "keep"},
        }
        ids = select_pubmedbert_plus_verified([pubmed, rescue, weak], matrix)
        self.assertEqual(ids, ["OUT-C0001", "OUT-C0002"])

    def test_comparison_filter_requires_comparator_cue(self):
        self.assertTrue(passes_label_filter(self.candidate("COM-C0001", "ComparisonGroup", "wait-list control")))
        self.assertFalse(passes_label_filter(self.candidate("COM-C0002", "ComparisonGroup", "students improved")))

    def test_sample_size_rejects_group_without_number_and_rescues_full_counts(self):
        bad_anchor = self.candidate("SAM-C0001", "SampleSize", "the intervention school", experts=["PubMedBERT"])
        rescue = self.candidate("SAM-C0002", "SampleSize", "196 from the intervention school, and 183 from the control school", start=40, experts=[])
        ids = select_pubmedbert_plus_verified([bad_anchor, rescue], {})
        self.assertEqual(ids, ["SAM-C0002"])

    def test_outcome_prefers_instrument_phrase_over_one_word_anchor(self):
        anchor = self.candidate("OUT-C0001", "Outcome", "mood", experts=["PubMedBERT"])
        rescue = self.candidate("OUT-C0002", "Outcome", "Mood and Feelings", start=40, experts=["ClinicalBERT"])
        ids = select_pubmedbert_plus_verified([anchor, rescue], {})
        self.assertEqual(ids, ["OUT-C0002"])

    def test_comparison_prefers_group_phrase_over_one_word_control(self):
        anchor = self.candidate("COM-C0001", "ComparisonGroup", "control", experts=["ClinicalBERT"])
        rescue = self.candidate("COM-C0002", "ComparisonGroup", "wait-list control group", start=40, experts=[])
        ids = select_pubmedbert_plus_verified([anchor, rescue], {})
        self.assertEqual(ids, ["COM-C0002"])

    def test_equal_expert_policy_keeps_multi_expert_candidate_without_pubmedbert_anchor(self):
        agreed = self.candidate(
            "INT-C0001",
            "Intervention",
            "mindfulness curriculum",
            experts=["SciBERT", "ClinicalBERT"],
        )
        single = self.candidate(
            "INT-C0002",
            "Intervention",
            "CBT skills training",
            start=50,
            experts=["PubMedBERT"],
        )
        ids = select_expert_consensus_verified([agreed, single], {})
        self.assertEqual(ids, ["INT-C0001"])

    def test_equal_expert_policy_uses_doctor_majority_for_single_expert_candidate(self):
        single = self.candidate(
            "OUT-C0001",
            "Outcome",
            "anxiety symptoms",
            experts=["ClinicalBERT"],
        )
        matrix = {"OUT-C0001": {"A": "keep", "B": "keep", "C": "reject"}}
        ids = select_expert_consensus_verified([single], matrix)
        self.assertEqual(ids, ["OUT-C0001"])

    def test_sample_size_filter_allows_ordinary_or_text(self):
        candidate = self.candidate("SAM-C0001", "SampleSize", "120 students or teachers", experts=["SciBERT"])
        self.assertTrue(passes_label_filter(candidate))


if __name__ == "__main__":
    unittest.main()
