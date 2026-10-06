import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.candidate_registry import CandidateRegistry
from Labelcentered.expert_models import RawCandidate, load_thresholds
from Labelcentered.settings import ExpertSpec, LABELS


def raw(label, text, start, end, expert="SciBERT", confidence=0.9):
    return RawCandidate(label, text, start, end, confidence, expert, 0.42, [confidence], "methods", {"expert": expert})


class CandidateRegistryTests(unittest.TestCase):
    def test_exact_candidate_construction_and_duplicate_merging(self):
        text = "=== METHODS ===\n63 participants joined. The wait-list control group continued usual care."
        start = text.index("63 participants")
        end = start + len("63 participants")
        registry = CandidateRegistry(text)
        registry.build_from_raw({
            "SciBERT": {"SampleSize": [raw("SampleSize", "63 participants", start, end, "SciBERT")]},
            "PubMedBERT": {"SampleSize": [raw("SampleSize", "63 participants", start, end, "PubMedBERT", 0.8)]},
        })
        candidates = registry.candidates_for_label("SampleSize")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].candidate_id, "SAM-C0001")
        self.assertEqual(candidates[0].proposed_by_experts, ["PubMedBERT", "SciBERT"])
        self.assertEqual(text[candidates[0].start:candidates[0].end], candidates[0].text)

    def test_label_specific_shared_bank_construction(self):
        text = "Outcome scores improved after therapy."
        start = text.index("Outcome scores")
        registry = CandidateRegistry(text)
        registry.build_from_raw({"SciBERT": {"Outcome": [raw("Outcome", "Outcome scores", start, start + 14)]}})
        self.assertEqual([c.candidate_id for c in registry.candidates_for_label("Outcome")], ["OUT-C0001"])
        self.assertEqual(registry.candidates_for_label("Participant"), [])

    def test_invalid_offset_rejection(self):
        text = "abc"
        registry = CandidateRegistry(text)
        candidate, reason = registry.add_candidate(label="Outcome", text="zzz", start=0, end=3)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "text_offset_mismatch")

    def test_long_candidate_with_pdf_hyphenation_is_retained(self):
        text = (
            "The log-transformed post-test scores were linearly regressed on "
            "the interven- tion effect and adjusted covariates."
        )
        registry = CandidateRegistry(text)
        candidate, reason = registry.add_candidate(
            label="StatisticalAnalysis",
            text=text,
            start=0,
            end=len(text),
        )
        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(candidate)

    def test_dangling_hyphen_fragment_is_rejected(self):
        text = "self- esteem"
        registry = CandidateRegistry(text)
        candidate, reason = registry.add_candidate(
            label="Outcome",
            text="self-",
            start=0,
            end=5,
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "broken_subword_fragment")

    def test_incomplete_sample_size_parenthesis_is_rejected(self):
        text = "(n=110 participants)"
        registry = CandidateRegistry(text)
        candidate, reason = registry.add_candidate(
            label="SampleSize",
            text="(n=110",
            start=0,
            end=6,
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "incomplete_sample_size_parenthesis")

    def test_incomplete_trailing_phrase_is_rejected(self):
        text = "frequency of past 6-month intake of"
        registry = CandidateRegistry(text)
        candidate, reason = registry.add_candidate(
            label="Outcome",
            text=text,
            start=0,
            end=len(text),
        )
        self.assertIsNone(candidate)
        self.assertEqual(reason, "incomplete_trailing_word")

    def test_short_statistical_fragment_is_rejected_but_t_test_is_allowed(self):
        text = "ibus t-test"
        registry = CandidateRegistry(text)
        fragment, fragment_reason = registry.add_candidate(
            label="StatisticalAnalysis",
            text="ibus",
            start=0,
            end=4,
        )
        method, method_reason = registry.add_candidate(
            label="StatisticalAnalysis",
            text="t-test",
            start=5,
            end=11,
        )
        self.assertIsNone(fragment)
        self.assertEqual(fragment_reason, "short_statistical_fragment")
        self.assertIsNotNone(method)
        self.assertEqual(method_reason, "accepted")


    def test_optional_boundary_normalization_repairs_sample_size(self):
        text = "The final analytic sample included 212 students in the intervention group."
        start = text.index("included")
        end = len(text) - 1
        registry = CandidateRegistry(text, normalize_boundaries=True)
        candidate, reason = registry.add_candidate(
            label="SampleSize",
            text=text[start:end],
            start=start,
            end=end,
        )
        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.text, "212 students")
        self.assertIn("boundary_normalization", candidate.provenance)

    def test_missing_threshold_failure_and_explicit_fallback(self):
        with TemporaryDirectory() as tmp:
            spec = ExpertSpec("SciBERT", "encoder", Path(tmp) / "missing")
            with self.assertRaises(FileNotFoundError):
                load_thresholds(spec, allow_default_thresholds=False)
            thresholds = load_thresholds(spec, allow_default_thresholds=True)
            self.assertEqual(set(thresholds), set(LABELS))
            self.assertTrue(all(value == 0.5 for value in thresholds.values()))


if __name__ == "__main__":
    unittest.main()
