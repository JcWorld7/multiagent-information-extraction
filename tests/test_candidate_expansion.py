import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.candidate_expansion import CandidateExpansionManager, deterministic_pre_expand_registry, segment_source_units
from Labelcentered.candidate_registry import CandidateRegistry


class CandidateExpansionTests(unittest.TestCase):
    def setUp(self):
        self.text = "The wait-list control group continued usual care. N = 63 participants completed the trial."
        self.registry = CandidateRegistry(self.text)
        start = self.text.index("wait-list control group")
        self.parent, _ = self.registry.add_candidate(label="ComparisonGroup", text="wait-list control group", start=start, end=start + len("wait-list control group"))
        self.units = segment_source_units(self.text)
        self.manager = CandidateExpansionManager(self.registry, self.units)

    def test_exact_quote_candidate_expansion(self):
        added = self.manager.handle_requests([{"label": "ComparisonGroup", "requested_operation": "exact_quote_proposal", "proposed_text": "usual care", "sentence_id": "S0001"}], proposed_by_agent="agent", generation_round=1)
        self.assertEqual(len(added), 1)
        self.assertEqual(self.text[added[0].start:added[0].end], added[0].text)

    def test_paraphrase_rejection(self):
        added = self.manager.handle_requests([{"label": "ComparisonGroup", "requested_operation": "exact_quote_proposal", "proposed_text": "regular healthcare"}], proposed_by_agent="agent", generation_round=1)
        self.assertEqual(added, [])
        self.assertEqual(self.manager.unresolved_proposals[-1]["reason"], "proposed_text_not_found")

    def test_duplicate_expansion_resolves_to_existing_candidate(self):
        added = self.manager.handle_requests([{"label": "ComparisonGroup", "requested_operation": "exact_quote_proposal", "proposed_text": "wait-list control group", "sentence_id": "S0001"}], proposed_by_agent="agent", generation_round=1)
        self.assertEqual(added, [])
        self.assertEqual(self.manager.unresolved_proposals, [])
        self.assertEqual(self.manager.last_request_outcomes[-1]["status"], "already_exists")
        self.assertEqual(
            self.manager.last_request_outcomes[-1]["candidate_id"],
            self.parent.candidate_id,
        )

    def test_repeated_resolved_request_is_suppressed(self):
        request = {
            "label": "ComparisonGroup",
            "requested_operation": "exact_quote_proposal",
            "proposed_text": "wait-list control group",
            "sentence_id": "S0001",
        }
        self.manager.handle_requests([request], proposed_by_agent="agent", generation_round=1)
        self.manager.handle_requests([request], proposed_by_agent="agent", generation_round=2)
        self.assertEqual(
            self.manager.last_request_outcomes[-1]["status"],
            "previously_resolved",
        )

    def test_repeated_unresolved_request_is_not_executed_twice(self):
        request = {
            "label": "ComparisonGroup",
            "requested_operation": "exact_quote_proposal",
            "proposed_text": "paraphrased comparator",
            "sentence_id": "S0001",
        }
        self.manager.handle_requests([request], proposed_by_agent="agent", generation_round=1)
        unresolved_count = len(self.manager.unresolved_proposals)
        self.manager.handle_requests([request], proposed_by_agent="agent", generation_round=2)
        self.assertEqual(len(self.manager.unresolved_proposals), unresolved_count)
        self.assertEqual(
            self.manager.last_request_outcomes[-1]["status"],
            "previously_unresolved",
        )

    def test_missing_sample_size_recovery(self):
        added = self.manager.handle_requests([{"label": "SampleSize", "requested_operation": "generate_number_centered_span"}], proposed_by_agent="agent", generation_round=1)
        self.assertTrue(any("63 participants" in candidate.text or "N = 63" in candidate.text for candidate in added))

    def test_sentence_expansion_without_parent_creates_full_sentence_candidate(self):
        added = self.manager.handle_requests(
            [{
                "label": "ComparisonGroup",
                "requested_operation": "expand_to_sentence_start",
                "sentence_id": "S0001",
            }],
            proposed_by_agent="ComparisonGroupMetaAgent",
            generation_round=1,
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].text, self.units[0].text)
        self.assertEqual(added[0].parent_candidate_id, None)
        self.assertEqual(added[0].generation_origin, "missing_label_recovery")
        self.assertEqual(self.text[added[0].start:added[0].end], added[0].text)

    def test_sentence_expansion_ignores_unknown_parent_when_sentence_is_valid(self):
        added = self.manager.handle_requests(
            [{
                "label": "ComparisonGroup",
                "requested_operation": "expand_to_sentence_end",
                "sentence_id": "S0001",
                "parent_candidate_id": "COM-C9999",
            }],
            proposed_by_agent="ComparisonGroupMetaAgent",
            generation_round=1,
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].text, self.units[0].text)
        self.assertIsNone(added[0].parent_candidate_id)
        self.assertEqual(
            added[0].provenance["ignored_parent_candidate_id"],
            "COM-C9999",
        )

    def test_sentence_expansion_rejects_unknown_sentence_without_parent(self):
        added = self.manager.handle_requests(
            [{
                "label": "ComparisonGroup",
                "requested_operation": "expand_to_sentence_start",
                "sentence_id": "S9999",
            }],
            proposed_by_agent="ComparisonGroupMetaAgent",
            generation_round=1,
        )
        self.assertEqual(added, [])
        self.assertEqual(self.manager.unresolved_proposals[-1]["reason"], "unknown_sentence_id")

    def test_sentence_expansion_normalizes_decorated_sentence_id(self):
        added = self.manager.handle_requests(
            [{
                "label": "ComparisonGroup",
                "requested_operation": "expand_to_sentence_end",
                "sentence_id": "S0001 0:49",
            }],
            proposed_by_agent="ComparisonGroupMetaAgent",
            generation_round=1,
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].text, self.units[0].text)

    def test_sentence_expansion_normalizes_underscore_sentence_id(self):
        added = self.manager.handle_requests(
            [{
                "label": "ComparisonGroup",
                "requested_operation": "expand_to_sentence_start",
                "sentence_id": "S0001_0",
            }],
            proposed_by_agent="ComparisonGroupMetaAgent",
            generation_round=1,
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].text, self.units[0].text)

    def test_sentence_expansion_resolves_source_offset(self):
        added = self.manager.handle_requests(
            [{
                "label": "SampleSize",
                "requested_operation": "expand_to_sentence_end",
                "sentence_id": str(self.units[1].start),
            }],
            proposed_by_agent="SampleSizeMetaAgent",
            generation_round=1,
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].text, self.units[1].text)

    def test_parent_dependent_split_still_requires_valid_parent(self):
        added = self.manager.handle_requests(
            [{
                "label": "ComparisonGroup",
                "requested_operation": "split_by_sentence",
                "parent_candidate_id": "COM-C9999",
            }],
            proposed_by_agent="ComparisonGroupMetaAgent",
            generation_round=1,
        )
        self.assertEqual(added, [])
        self.assertEqual(self.manager.unresolved_proposals[-1]["reason"], "unknown_parent_candidate_id")

    def test_labels_with_no_spans_empty_arrays(self):
        self.assertEqual(self.registry.candidates_for_label("Outcome"), [])

    def test_deterministic_pre_expansion_adds_exact_core_label_candidates(self):
        text = (
            "Middle school students aged 12 to 14 participated. "
            "The mindfulness curriculum included 8 weekly sessions. "
            "Outcomes included anxiety symptoms and the SDQ scale. "
            "N = 63 participants completed the trial."
        )
        registry = CandidateRegistry(text)
        added = deterministic_pre_expand_registry(registry, segment_source_units(text))
        self.assertTrue(added)
        by_label = {label: [candidate.text for candidate in registry.candidates_for_label(label)] for label in ["Participant", "Intervention", "Outcome", "SampleSize"]}
        self.assertTrue(any("students" in text for text in by_label["Participant"]))
        self.assertTrue(any("mindfulness curriculum" in text for text in by_label["Intervention"]))
        self.assertTrue(any("anxiety symptoms" in text for text in by_label["Outcome"]))
        self.assertTrue(any("N = 63" in text for text in by_label["SampleSize"]))
        for candidate in added:
            self.assertEqual(text[candidate.start:candidate.end], candidate.text)
            self.assertEqual(candidate.generation_origin, "deterministic_pre_expansion")

    def test_deterministic_pre_expansion_adds_boundary_alternative_inside_broad_candidate(self):
        text = "The outcome sentence included anxiety symptoms measured at posttest and unrelated words."
        registry = CandidateRegistry(text)
        candidate, reason = registry.add_candidate(
            label="Outcome",
            text=text,
            start=0,
            end=len(text),
        )
        self.assertEqual(reason, "accepted")
        added = deterministic_pre_expand_registry(registry, segment_source_units(text), labels=["Outcome"])
        self.assertTrue(any(candidate.text == "anxiety symptoms" for candidate in added))


if __name__ == "__main__":
    unittest.main()
