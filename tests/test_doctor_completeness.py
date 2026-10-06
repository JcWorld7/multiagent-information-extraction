import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.consistency import normalize_review, validate_review_partition
from Labelcentered.candidate_registry import CandidateRegistry
from Labelcentered.candidate_expansion import segment_source_units
from Labelcentered.label_doctor import (
    adaptive_candidate_batches,
    batched,
    bounded_excerpt,
    candidate_context,
    render_doctor_source_units,
    merge_review_batches,
    source_units_for_doctor_batch,
)
from Labelcentered.settings import PipelineSettings


class DoctorCompletenessTests(unittest.TestCase):
    def test_complete_keep_reject_decisions(self):
        review = normalize_review("SciBERT_Outcome_Doctor", {
            "decisions": [
                {"candidate_id": "OUT-C0001", "decision": "keep"},
                {"candidate_id": "OUT-C0002", "decision": "reject"},
            ]
        }, ["OUT-C0001", "OUT-C0002"])
        self.assertTrue(review["review_complete"])

    def test_missing_decision_retry_merge(self):
        first = normalize_review("SciBERT_Outcome_Doctor", {"keep_ids": ["OUT-C0001"], "reject_ids": []}, ["OUT-C0001", "OUT-C0002"])
        second = normalize_review("SciBERT_Outcome_Doctor", {"keep_ids": [], "reject_ids": ["OUT-C0002"]}, ["OUT-C0002"])
        merged = merge_review_batches("SciBERT_Outcome_Doctor", ["OUT-C0001", "OUT-C0002"], [first, second])
        self.assertTrue(merged["review_complete"])
        self.assertEqual(merged["decisions_by_candidate"]["OUT-C0002"], "reject")

    def test_empty_review_rejection(self):
        validation = validate_review_partition([], [], ["OUT-C0001"])
        self.assertFalse(validation["valid"])
        self.assertTrue(validation["empty_review"])

    def test_unknown_id_rejection(self):
        validation = validate_review_partition(["BAD"], [], ["OUT-C0001"])
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["unknown_ids"], ["BAD"])

    def test_keep_reject_overlap_rejection(self):
        validation = validate_review_partition(["OUT-C0001"], ["OUT-C0001"], ["OUT-C0001"])
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["overlap_ids"], ["OUT-C0001"])

    def test_all_three_doctors_receive_same_candidate_ids(self):
        candidate_ids = ["OUT-C0001", "OUT-C0002", "OUT-C0003"]
        assigned = {
            "SciBERT_Outcome_Doctor": [cid for batch in batched(candidate_ids, 2) for cid in batch],
            "PubMedBERT_Outcome_Doctor": [cid for batch in batched(candidate_ids, 2) for cid in batch],
            "ClinicalBERT_Outcome_Doctor": [cid for batch in batched(candidate_ids, 2) for cid in batch],
        }
        self.assertEqual(len({tuple(ids) for ids in assigned.values()}), 1)
        self.assertEqual(assigned["SciBERT_Outcome_Doctor"], candidate_ids)

    def test_long_candidate_context_is_bounded(self):
        source_text = "before " + ("long intervention text " * 2000) + " after"
        start = len("before ")
        end = source_text.index(" after")
        registry = CandidateRegistry(source_text)
        candidate, _ = registry.add_candidate(
            label="Intervention",
            text=source_text[start:end],
            start=start,
            end=end,
        )
        context = candidate_context(source_text, candidate)
        self.assertLess(len(context), 1200)
        self.assertIn("[DISPLAY TRUNCATED]", context)

    def test_adaptive_batches_split_large_rendered_candidates(self):
        source_text = " ".join(["intervention"] * 6000)
        registry = CandidateRegistry(source_text)
        candidates = []
        for index in range(6):
            start = index * 4000
            end = min(len(source_text), start + 3500)
            candidate, _ = registry.add_candidate(
                label="Intervention",
                text=source_text[start:end],
                start=start,
                end=end,
            )
            candidates.append(candidate)
        settings = PipelineSettings(
            doctor_batch_size=10,
            llm_context_window=3000,
            llm_max_tokens=1000,
        )
        batches = list(adaptive_candidate_batches(candidates, source_text, settings))
        self.assertGreater(len(batches), 1)
        self.assertEqual(
            [candidate.candidate_id for batch in batches for candidate in batch],
            [candidate.candidate_id for candidate in candidates],
        )


    def test_doctor_source_units_include_label_cue_rescue_context(self):
        source_text = (
            "The vague outcome candidate was introduced here. "
            "Later, anxiety symptoms were measured using the SDQ scale."
        )
        registry = CandidateRegistry(source_text)
        start = source_text.index("vague outcome candidate")
        candidate, _ = registry.add_candidate(
            label="Outcome",
            text="vague outcome candidate",
            start=start,
            end=start + len("vague outcome candidate"),
        )
        units = segment_source_units(source_text)
        selected = source_units_for_doctor_batch("Outcome", [candidate], units, max_units=3)
        rendered = render_doctor_source_units(selected)
        self.assertIn("S0001", rendered)
        self.assertIn("S0002", rendered)
        self.assertIn("anxiety symptoms", rendered)

    def test_bounded_excerpt_preserves_both_boundaries(self):
        excerpt = bounded_excerpt("BEGIN" + ("x" * 1000) + "END", 100)
        self.assertTrue(excerpt.startswith("BEGIN"))
        self.assertTrue(excerpt.endswith("END"))


if __name__ == "__main__":
    unittest.main()
