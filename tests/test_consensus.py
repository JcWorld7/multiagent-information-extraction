import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Labelcentered.consistency import cohen_kappa, consistency_metrics, evaluate_label_consensus, fleiss_kappa
from Labelcentered.candidate_expansion import CandidateExpansionManager, segment_source_units
from Labelcentered.candidate_registry import CandidateRegistry
from Labelcentered.consultation import (
    prune_contained_strict_candidates,
    run_one_label_consultation,
)
from Labelcentered.label_meta_agent import (
    meta_candidate_char_budget,
    meta_source_unit_char_budget,
    meta_source_unit_limit,
    prune_redundant_selected_candidates,
    render_meta_candidates,
    source_units_for_candidates,
    valid_consensus_proof,
    valid_source_search_requests,
)
from Labelcentered.llm_client import LLMClient
from Labelcentered.rag_retriever import GuidelineRetriever
from Labelcentered.settings import PipelineSettings


def review(role, decisions):
    return {
        "agent_role": role,
        "decisions_by_candidate": decisions,
        "review_complete": True,
        "validation": {"unknown_ids": [], "overlap_ids": [], "missing_ids": []},
    }


def proof(candidate_id, quote, sentence_id="S0001"):
    return {
        "candidate_id": candidate_id,
        "evidence_sentence_id": sentence_id,
        "evidence_quote": quote,
        "label_fit_rationale": "The quote expresses the requested label.",
        "boundary_rationale": "The selected span is copied from the displayed source.",
        "why_not_other_labels": {"Intervention": "not an intervention"},
        "counterevidence_checked": True,
    }


class ConsensusTests(unittest.TestCase):
    def test_unanimous_keep_consensus(self):
        ids = ["OUT-C0001"]
        reviews = [review("A", {"OUT-C0001": "keep"}), review("B", {"OUT-C0001": "keep"}), review("C", {"OUT-C0001": "keep"})]
        result = evaluate_label_consensus(reviews, ids)
        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["unanimously_selected"], ids)

    def test_majority_not_unanimous(self):
        ids = ["OUT-C0001"]
        reviews = [review("A", {"OUT-C0001": "keep"}), review("B", {"OUT-C0001": "keep"}), review("C", {"OUT-C0001": "reject"})]
        result = evaluate_label_consensus(reviews, ids)
        self.assertFalse(result["consensus_reached"])
        self.assertEqual(result["majority_selected"], ids)

    def test_empty_selection_is_not_consensus(self):
        reviews = [review("A", {}), review("B", {}), review("C", {})]
        result = evaluate_label_consensus(reviews, [])
        self.assertFalse(result["consensus_reached"])
        self.assertFalse(result["has_selected_candidates"])

    def test_maximum_round_unresolved_reason(self):
        ids = ["OUT-C0001"]
        reviews = [review("A", {"OUT-C0001": "keep"}), review("B", {"OUT-C0001": "keep"}), review("C", {"OUT-C0001": "reject"})]
        result = evaluate_label_consensus(reviews, ids)
        reason = "strict_unanimous_candidate_consensus" if result["consensus_reached"] else "maximum_rounds_reached_with_unresolved_decisions"
        self.assertEqual(reason, "maximum_rounds_reached_with_unresolved_decisions")

    def test_pairwise_cohen_fleiss_and_vote_matrix(self):
        ids = ["C1", "C2"]
        reviews = [review("A", {"C1": "keep", "C2": "reject"}), review("B", {"C1": "keep", "C2": "keep"}), review("C", {"C1": "keep", "C2": "reject"})]
        metrics = consistency_metrics(reviews, ids)
        self.assertEqual(metrics["vote_matrix"]["C1"]["A"], "keep")
        self.assertEqual(metrics["n_candidates_reviewed_by_all_three"], 2)
        self.assertIsNotNone(cohen_kappa(["keep", "reject"], ["keep", "keep"]))
        self.assertIsNotNone(fleiss_kappa([["keep", "keep", "keep"], ["reject", "keep", "reject"]]))

    def test_redundant_overlapping_selections_are_pruned(self):
        text = "Intervention The programme consisted of twelve sessions."
        registry = CandidateRegistry(text)
        first, _ = registry.add_candidate(
            label="Intervention",
            text=text,
            start=0,
            end=len(text),
        )
        second_text = text[len("Intervention "):]
        second, _ = registry.add_candidate(
            label="Intervention",
            text=second_text,
            start=len("Intervention "),
            end=len(text),
        )
        kept, removed = prune_redundant_selected_candidates(
            [first, second],
            [first.candidate_id, second.candidate_id],
        )
        self.assertEqual(kept, [second.candidate_id])
        self.assertEqual(removed, [first.candidate_id])

    def test_empty_statistical_bank_retrieves_late_label_cue_units(self):
        text = (
            "Participants entered the trial. The intervention was delivered. "
            "Later methods were described. Stata was used for all analyses."
        )
        units = segment_source_units(text)
        selected = source_units_for_candidates(
            "StatisticalAnalysis",
            [],
            units,
        )
        self.assertTrue(any("Stata" in unit.text for unit in selected))


    def test_candidate_near_units_also_include_label_cue_rescue_units(self):
        text = (
            "The vague outcome candidate was mentioned here. "
            "Later, the primary outcome was depressive symptoms measured using CES-D."
        )
        registry = CandidateRegistry(text)
        start = text.index("vague outcome candidate")
        candidate, _ = registry.add_candidate(
            label="Outcome",
            text="vague outcome candidate",
            start=start,
            end=start + len("vague outcome candidate"),
        )
        units = segment_source_units(text)
        selected = source_units_for_candidates("Outcome", [candidate], units)
        self.assertTrue(any("depressive symptoms" in unit.text for unit in selected))

    def test_consensus_proof_requires_quote_in_displayed_source_unit(self):
        text = "The primary outcome was depressive symptoms measured using CES-D."
        units = segment_source_units(text)
        valid = valid_consensus_proof(
            [{
                "candidate_id": "OUT-C0001",
                "evidence_sentence_id": "S0001",
                "evidence_quote": "depressive symptoms measured using CES-D",
                "label_fit_rationale": "Names a measured construct.",
                "boundary_rationale": "Grounded in the sentence.",
                "why_not_other_labels": {"Intervention": "not a program"},
                "counterevidence_checked": True,
            }],
            ["OUT-C0001"],
            units,
        )
        invalid = valid_consensus_proof(
            [{
                "candidate_id": "OUT-C0001",
                "evidence_sentence_id": "S0001",
                "evidence_quote": "hallucinated quote",
                "label_fit_rationale": "",
                "boundary_rationale": "",
                "why_not_other_labels": {},
                "counterevidence_checked": False,
            }],
            ["OUT-C0001"],
            units,
        )
        self.assertEqual(len(valid), 1)
        self.assertEqual(invalid, [])

    def test_source_search_requests_are_label_and_operation_bounded(self):
        rows = valid_source_search_requests(
            [
                {"label": "Outcome", "reason": "needs rescue", "search_cues": ["outcome"], "requested_operation": "repair_boundary"},
                {"label": "SampleSize", "reason": "wrong label", "search_cues": ["n="], "requested_operation": "repair_boundary"},
                {"label": "Outcome", "reason": "bad op", "search_cues": [], "requested_operation": "invent_offsets"},
            ],
            "Outcome",
            allow_expansion=True,
        )
        self.assertEqual(rows, [{"label": "Outcome", "reason": "needs rescue", "search_cues": ["outcome"], "requested_operation": "repair_boundary"}])

    def test_meta_candidate_rendering_has_fixed_total_budget(self):
        text = " ".join(["outcome"] * 10000)
        registry = CandidateRegistry(text)
        candidates = []
        for index in range(40):
            start = index * 1000
            end = min(len(text), start + 900)
            candidate, _ = registry.add_candidate(
                label="Outcome",
                text=text[start:end],
                start=start,
                end=end,
            )
            candidates.append(candidate)
        rendered = render_meta_candidates(text, candidates, total_char_budget=12000)
        self.assertLess(len(rendered), 18000)
        for candidate in candidates:
            self.assertIn(candidate.candidate_id, rendered)

    def test_meta_prompt_budgets_are_conservative_for_8k_and_16k_contexts(self):
        small = PipelineSettings(llm_context_window=8192)
        default = PipelineSettings(llm_context_window=16384)
        self.assertLessEqual(meta_candidate_char_budget(small), 8000)
        self.assertLessEqual(meta_source_unit_limit(small), 6)
        self.assertLessEqual(meta_source_unit_char_budget(small), 300)
        self.assertLessEqual(meta_candidate_char_budget(default), 12000)
        self.assertLessEqual(meta_source_unit_limit(default), 8)
        self.assertLessEqual(meta_source_unit_char_budget(default), 360)

    def test_nested_sample_size_selection_prefers_complete_outer_span(self):
        text = "The 378 participants were assigned to groups (n=126 and n=124)."
        registry = CandidateRegistry(text)
        outer, _ = registry.add_candidate(
            label="SampleSize",
            text=text,
            start=0,
            end=len(text),
        )
        inner_text = "groups (n=126 and n=124)"
        inner_start = text.index(inner_text)
        inner, _ = registry.add_candidate(
            label="SampleSize",
            text=inner_text,
            start=inner_start,
            end=inner_start + len(inner_text),
        )
        kept, removed = prune_contained_strict_candidates(
            [outer, inner],
            [outer.candidate_id, inner.candidate_id],
        )
        self.assertEqual(kept, [outer.candidate_id])
        self.assertEqual(removed, [inner.candidate_id])

    def test_disagreement_expands_registry_then_all_doctors_review_new_candidate(self):
        source_text = "The broad outcome phrase included noise. Exact outcome score improved significantly."
        registry = CandidateRegistry(source_text)
        initial_start = source_text.index("broad outcome phrase included noise")
        initial, reason = registry.add_candidate(
            label="Outcome",
            text="broad outcome phrase included noise",
            start=initial_start,
            end=initial_start + len("broad outcome phrase included noise"),
            proposed_by_experts=["SciBERT"],
            confidence_by_expert={"SciBERT": 0.91},
            threshold_by_expert={"SciBERT": 0.4},
        )
        self.assertEqual(reason, "accepted")
        self.assertEqual(initial.candidate_id, "OUT-C0001")

        def doctor_response(decisions):
            return {
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "decision": decision,
                        "issue_type": "acceptable" if decision == "keep" else "boundary_too_long",
                        "justification": "mocked review",
                    }
                    for candidate_id, decision in decisions.items()
                ],
                "expansion_requests": [],
                "missing_label": False,
                "comment": "",
            }

        meta_expansion = {
            "selected_candidate_ids": [],
            "rejected_candidate_ids": ["OUT-C0001"],
            "expansion_requests": [
                {
                    "label": "Outcome",
                    "parent_candidate_id": "OUT-C0001",
                    "requested_operation": "exact_quote_proposal",
                    "proposed_text": "Exact outcome score",
                    "sentence_id": "S0002",
                    "justification": "The existing candidate has poor boundaries.",
                }
            ],
            "unresolved_candidate_ids": ["OUT-C0001"],
            "rationale": "Request an exact replacement quotation.",
        }
        meta_consensus = {
            "selected_candidate_ids": ["OUT-C0002"],
            "rejected_candidate_ids": ["OUT-C0001"],
            "expansion_requests": [],
            "unresolved_candidate_ids": [],
            "consensus_proof": [proof("OUT-C0002", "Exact outcome score", "S0002")],
            "source_search_requests": [],
            "rationale": "All doctors kept the verified replacement candidate.",
        }
        mock_responses = [
            doctor_response({"OUT-C0001": "reject"}),
            doctor_response({"OUT-C0001": "keep"}),
            doctor_response({"OUT-C0001": "reject"}),
            meta_expansion,
            doctor_response({"OUT-C0001": "reject", "OUT-C0002": "keep"}),
            doctor_response({"OUT-C0001": "reject", "OUT-C0002": "keep"}),
            doctor_response({"OUT-C0001": "reject", "OUT-C0002": "keep"}),
            meta_consensus,
        ]
        settings = PipelineSettings(max_rounds=3, doctor_batch_size=10)
        units = segment_source_units(source_text)
        result = run_one_label_consultation(
            label="Outcome",
            source_text=source_text,
            source_units=units,
            registry=registry,
            expansion_manager=CandidateExpansionManager(registry, units),
            settings=settings,
            llm_client=LLMClient(settings, mock_responses=mock_responses),
            retriever=GuidelineRetriever(),
        )

        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["termination_reason"], "strict_unanimous_candidate_consensus")
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["rounds"][0]["new_candidate_ids"], ["OUT-C0002"])
        self.assertTrue(result["rounds"][0]["new_candidates_require_review"])
        self.assertIn("OUT-C0002", result["rounds"][1]["candidate_ids_reviewed"])
        self.assertEqual(result["final_candidate_ids"], ["OUT-C0002"])
        added = registry.by_id["OUT-C0002"]
        self.assertEqual(added.text, "Exact outcome score")
        self.assertEqual(source_text[added.start:added.end], added.text)

    def test_majority_selected_candidate_is_not_put_in_final_ids(self):
        source_text = "Outcome score improved."
        registry = CandidateRegistry(source_text)
        candidate, _ = registry.add_candidate(
            label="Outcome",
            text="Outcome score",
            start=0,
            end=len("Outcome score"),
        )

        def doctor(decision):
            return {
                "decisions": [{
                    "candidate_id": candidate.candidate_id,
                    "decision": decision,
                    "issue_type": "acceptable",
                    "justification": "mock",
                }],
                "expansion_requests": [],
                "missing_label": False,
                "comment": "",
            }

        meta = {
            "selected_candidate_ids": [candidate.candidate_id],
            "rejected_candidate_ids": [],
            "expansion_requests": [],
            "unresolved_candidate_ids": [candidate.candidate_id],
            "rationale": "majority only",
        }
        settings = PipelineSettings(max_rounds=1)
        units = segment_source_units(source_text)
        result = run_one_label_consultation(
            label="Outcome",
            source_text=source_text,
            source_units=units,
            registry=registry,
            expansion_manager=CandidateExpansionManager(registry, units),
            settings=settings,
            llm_client=LLMClient(
                settings,
                mock_responses=[doctor("keep"), doctor("keep"), doctor("reject"), meta],
            ),
            retriever=GuidelineRetriever(),
        )
        self.assertFalse(result["consensus_reached"])
        self.assertEqual(result["final_candidate_ids"], [])
        self.assertEqual(result["recommended_candidate_ids"], [candidate.candidate_id])

    def test_duplicate_expansion_request_does_not_block_consensus(self):
        source_text = "Outcome score improved."
        registry = CandidateRegistry(source_text)
        candidate, _ = registry.add_candidate(
            label="Outcome",
            text="Outcome score",
            start=0,
            end=len("Outcome score"),
        )
        doctor = {
            "decisions": [{
                "candidate_id": candidate.candidate_id,
                "decision": "keep",
                "issue_type": "acceptable",
                "justification": "mock",
            }],
            "expansion_requests": [],
            "missing_label": False,
            "comment": "",
        }
        meta = {
            "selected_candidate_ids": [candidate.candidate_id],
            "rejected_candidate_ids": [],
            "expansion_requests": [{
                "label": "Outcome",
                "requested_operation": "exact_quote_proposal",
                "proposed_text": "Outcome score",
                "sentence_id": "S0001",
                "justification": "duplicate request",
            }],
            "unresolved_candidate_ids": [],
            "consensus_proof": [proof(candidate.candidate_id, "Outcome score")],
            "source_search_requests": [],
            "rationale": "select existing exact span",
        }
        settings = PipelineSettings(max_rounds=1)
        units = segment_source_units(source_text)
        result = run_one_label_consultation(
            label="Outcome",
            source_text=source_text,
            source_units=units,
            registry=registry,
            expansion_manager=CandidateExpansionManager(registry, units),
            settings=settings,
            llm_client=LLMClient(settings, mock_responses=[doctor, doctor, doctor, meta]),
            retriever=GuidelineRetriever(),
        )
        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["final_candidate_ids"], [candidate.candidate_id])
        self.assertEqual(
            result["rounds"][0]["proposal_outcomes"][0]["status"],
            "already_exists",
        )

    def test_best_unanimous_state_survives_weaker_last_round(self):
        source_text = "Outcome score improved."
        registry = CandidateRegistry(source_text)
        candidate, _ = registry.add_candidate(
            label="Outcome",
            text="Outcome score",
            start=0,
            end=len("Outcome score"),
        )

        def doctor(decision):
            return {
                "decisions": [{
                    "candidate_id": candidate.candidate_id,
                    "decision": decision,
                    "issue_type": "acceptable",
                    "justification": "mock",
                }],
                "expansion_requests": [],
                "missing_label": False,
                "comment": "",
            }

        first_meta = {
            "selected_candidate_ids": [candidate.candidate_id],
            "rejected_candidate_ids": [],
            "expansion_requests": [{
                "label": "Outcome",
                "requested_operation": "exact_quote_proposal",
                "proposed_text": "paraphrased outcome",
                "sentence_id": "S0001",
                "justification": "mock unresolved request",
            }],
            "unresolved_candidate_ids": [],
            "rationale": "unanimous but unresolved proposal",
        }
        second_meta = {
            "selected_candidate_ids": [candidate.candidate_id],
            "rejected_candidate_ids": [],
            "expansion_requests": [],
            "unresolved_candidate_ids": [candidate.candidate_id],
            "rationale": "weaker final round",
        }
        settings = PipelineSettings(max_rounds=2)
        units = segment_source_units(source_text)
        result = run_one_label_consultation(
            label="Outcome",
            source_text=source_text,
            source_units=units,
            registry=registry,
            expansion_manager=CandidateExpansionManager(registry, units),
            settings=settings,
            llm_client=LLMClient(
                settings,
                mock_responses=[
                    doctor("keep"),
                    doctor("keep"),
                    doctor("keep"),
                    first_meta,
                    doctor("keep"),
                    doctor("keep"),
                    doctor("reject"),
                    second_meta,
                ],
            ),
            retriever=GuidelineRetriever(),
        )
        self.assertFalse(result["consensus_reached"])
        self.assertEqual(result["final_candidate_ids"], [candidate.candidate_id])
        self.assertEqual(result["best_strict_selection_round"], 1)
        self.assertTrue(
            all(
                vote == "keep"
                for vote in result["candidate_vote_matrix"][candidate.candidate_id].values()
            )
        )

    def test_final_round_expansion_receives_review_only_extension(self):
        source_text = "Broad outcome. Exact outcome score improved."
        registry = CandidateRegistry(source_text)
        broad_start = source_text.index("Broad outcome")
        broad, _ = registry.add_candidate(
            label="Outcome",
            text="Broad outcome",
            start=broad_start,
            end=broad_start + len("Broad outcome"),
        )

        def doctor(decisions):
            return {
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "decision": decision,
                        "issue_type": "acceptable",
                        "justification": "mock",
                    }
                    for candidate_id, decision in decisions.items()
                ],
                "expansion_requests": [],
                "missing_label": False,
                "comment": "",
            }

        first_meta = {
            "selected_candidate_ids": [],
            "rejected_candidate_ids": [broad.candidate_id],
            "expansion_requests": [{
                "label": "Outcome",
                "requested_operation": "exact_quote_proposal",
                "proposed_text": "Exact outcome score",
                "sentence_id": "S0002",
                "justification": "repair",
            }],
            "unresolved_candidate_ids": [broad.candidate_id],
            "rationale": "append replacement",
        }
        final_meta = {
            "selected_candidate_ids": ["OUT-C0002"],
            "rejected_candidate_ids": [broad.candidate_id],
            "expansion_requests": [],
            "unresolved_candidate_ids": [],
            "consensus_proof": [proof("OUT-C0002", "Exact outcome score", "S0002")],
            "source_search_requests": [],
            "rationale": "reviewed replacement",
        }
        settings = PipelineSettings(max_rounds=1)
        units = segment_source_units(source_text)
        result = run_one_label_consultation(
            label="Outcome",
            source_text=source_text,
            source_units=units,
            registry=registry,
            expansion_manager=CandidateExpansionManager(registry, units),
            settings=settings,
            llm_client=LLMClient(
                settings,
                mock_responses=[
                    doctor({broad.candidate_id: "reject"}),
                    doctor({broad.candidate_id: "reject"}),
                    doctor({broad.candidate_id: "reject"}),
                    first_meta,
                    doctor({broad.candidate_id: "reject", "OUT-C0002": "keep"}),
                    doctor({broad.candidate_id: "reject", "OUT-C0002": "keep"}),
                    doctor({broad.candidate_id: "reject", "OUT-C0002": "keep"}),
                    final_meta,
                ],
            ),
            retriever=GuidelineRetriever(),
        )
        self.assertEqual(len(result["rounds"]), 2)
        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["final_candidate_ids"], ["OUT-C0002"])

    def test_doctor_expansion_request_appends_candidate_for_next_round(self):
        source_text = "Broad outcome wording was noisy. Exact anxiety score improved."
        registry = CandidateRegistry(source_text)
        broad_start = source_text.index("Broad outcome wording")
        broad, _ = registry.add_candidate(
            label="Outcome",
            text="Broad outcome wording",
            start=broad_start,
            end=broad_start + len("Broad outcome wording"),
        )

        def doctor(decisions, expansion_requests=None):
            return {
                "decisions": [
                    {
                        "candidate_id": candidate_id,
                        "decision": decision,
                        "issue_type": "acceptable" if decision == "keep" else "boundary_too_long",
                        "justification": "mock",
                    }
                    for candidate_id, decision in decisions.items()
                ],
                "expansion_requests": expansion_requests or [],
                "missing_label": False,
                "comment": "",
            }

        first_meta = {
            "selected_candidate_ids": [],
            "rejected_candidate_ids": [broad.candidate_id],
            "expansion_requests": [],
            "unresolved_candidate_ids": [broad.candidate_id],
            "consensus_proof": [],
            "source_search_requests": [],
            "rationale": "doctor requested a better quote",
        }
        final_meta = {
            "selected_candidate_ids": ["OUT-C0002"],
            "rejected_candidate_ids": [broad.candidate_id],
            "expansion_requests": [],
            "unresolved_candidate_ids": [],
            "consensus_proof": [proof("OUT-C0002", "Exact anxiety score", "S0002")],
            "source_search_requests": [],
            "rationale": "reviewed doctor proposal",
        }
        request = {
            "label": "Outcome",
            "requested_operation": "exact_quote_proposal",
            "proposed_text": "Exact anxiety score",
            "sentence_id": "S0002",
            "justification": "better complete phrase",
        }
        settings = PipelineSettings(max_rounds=2)
        units = segment_source_units(source_text)
        result = run_one_label_consultation(
            label="Outcome",
            source_text=source_text,
            source_units=units,
            registry=registry,
            expansion_manager=CandidateExpansionManager(registry, units),
            settings=settings,
            llm_client=LLMClient(
                settings,
                mock_responses=[
                    doctor({broad.candidate_id: "reject"}, [request]),
                    doctor({broad.candidate_id: "reject"}),
                    doctor({broad.candidate_id: "reject"}),
                    first_meta,
                    doctor({broad.candidate_id: "reject", "OUT-C0002": "keep"}),
                    doctor({broad.candidate_id: "reject", "OUT-C0002": "keep"}),
                    doctor({broad.candidate_id: "reject", "OUT-C0002": "keep"}),
                    final_meta,
                ],
            ),
            retriever=GuidelineRetriever(),
        )
        self.assertEqual(result["rounds"][0]["doctor_expansion_requests"], [request])
        self.assertEqual(result["rounds"][0]["new_candidate_ids"], ["OUT-C0002"])
        self.assertIn("OUT-C0002", result["rounds"][1]["candidate_ids_reviewed"])
        self.assertEqual(result["final_candidate_ids"], ["OUT-C0002"])



if __name__ == "__main__":
    unittest.main()
