"""Label-centered consultation orchestration."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from Labelcentered.calibration import load_calibration_config, select_calibrated_candidate_ids
from Labelcentered.candidate_cleaning import boundary_quality_issue
from Labelcentered.candidate_expansion import CandidateExpansionManager, SourceUnit, deterministic_pre_expand_registry, segment_source_units
from Labelcentered.candidate_registry import Candidate, CandidateRegistry
from Labelcentered.consistency import consistency_metrics, evaluate_label_consensus
from Labelcentered.global_meta_agent import GlobalMetaAgent
from Labelcentered.label_doctor import LabelDoctor
from Labelcentered.label_meta_agent import LabelMetaAgent
from Labelcentered.llm_client import LLMClient
from Labelcentered.rag_retriever import GuidelineRetriever
from Labelcentered.selection_policy import passes_label_filter, select_expert_consensus_verified, select_pubmedbert_plus_verified
from Labelcentered.settings import DOCTOR_NAMES, LABELS, PipelineSettings
from Labelcentered.summary_agent import SummaryAgent


def run_labelcentered_consultation(
    *,
    source_text: str,
    raw_candidates_by_expert_and_label: Mapping[str, Mapping[str, Any]],
    settings: PipelineSettings,
    llm_client: LLMClient,
    classifier_only: bool = False,
) -> Dict[str, Any]:
    selection_policy = getattr(settings, "final_selection_policy", "expert_consensus_verified")
    calibration_config = None
    if selection_policy == "calibrated":
        config_path = getattr(settings, "calibration_config_path", None)
        if not config_path:
            raise ValueError("--calibration_config is required when --final_selection_policy calibrated")
        calibration_config = load_calibration_config(config_path)
    registry = CandidateRegistry(source_text, normalize_candidate_boundaries=settings.normalize_candidate_boundaries)
    registry.build_from_raw(raw_candidates_by_expert_and_label)
    source_units = segment_source_units(source_text)
    deterministic_pre_expansion_candidates = []
    if settings.deterministic_pre_expansion:
        deterministic_pre_expansion_candidates = deterministic_pre_expand_registry(registry, source_units)
    expansion_manager = CandidateExpansionManager(registry, source_units)

    if classifier_only:
        final_ids_by_label = {label: [candidate.candidate_id for candidate in registry.candidates_for_label(label)] for label in LABELS}
        global_decision = GlobalMetaAgent(settings).assemble(registry, final_ids_by_label, {})
        return {
            "source_units": [unit.to_dict() for unit in source_units],
            "deterministic_pre_expansion_candidate_ids": [candidate.candidate_id for candidate in deterministic_pre_expansion_candidates],
            "cleaning_log": registry.cleaning_log,
            "label_candidate_registries": registry.registries_json(),
            "registry_expansion_history": [],
            "doctor_reviews_by_label": {label: [] for label in LABELS},
            "review_completeness_by_label": {label: None for label in LABELS},
            "consistency_metrics_by_label": {label: None for label in LABELS},
            "candidate_vote_matrices_by_label": {label: {} for label in LABELS},
            "label_meta_decisions": {},
            "label_consensus_status": {label: None for label in LABELS},
            "label_termination_reasons": {label: "classifier_only" for label in LABELS},
            "strict_global_meta_decision": global_decision,
            "global_meta_decision": global_decision,
            "final_selection_policy": selection_policy,
            "final_candidate_ids_by_label": final_ids_by_label,
            "recommended_candidate_ids_by_label": final_ids_by_label,
            "final_spans_by_label": global_decision["final_spans_by_label"],
            "final_spans": global_decision["final_spans"],
            "missing_labels": [label for label in LABELS if not final_ids_by_label[label]],
            "unresolved_candidates": [],
            "cross_label_conflicts": global_decision["cross_label_conflicts"],
            "overall_consensus_reached": None,
        }

    retriever = GuidelineRetriever()
    doctor_reviews_by_label: Dict[str, Any] = {}
    consistency_by_label: Dict[str, Any] = {}
    vote_matrices: Dict[str, Any] = {}
    label_meta_decisions: Dict[str, Any] = {}
    latest_label_meta_decisions: Dict[str, Any] = {}
    consensus_status: Dict[str, bool] = {}
    termination_reasons: Dict[str, str] = {}
    final_ids_by_label: Dict[str, list[str]] = {label: [] for label in LABELS}
    recommended_ids_by_label: Dict[str, list[str]] = {label: [] for label in LABELS}
    unresolved_candidates = []

    rounds_by_label: Dict[str, Any] = {}
    for label in LABELS:
        label_result = run_one_label_consultation(
            label=label,
            source_text=source_text,
            source_units=source_units,
            registry=registry,
            expansion_manager=expansion_manager,
            settings=settings,
            llm_client=llm_client,
            retriever=retriever,
        )
        doctor_reviews_by_label[label] = label_result["doctor_reviews"]
        consistency_by_label[label] = label_result["consistency_metrics"]
        vote_matrices[label] = label_result["candidate_vote_matrix"]
        label_meta_decisions[label] = label_result["label_meta_decision"]
        latest_label_meta_decisions[label] = label_result["latest_label_meta_decision"]
        consensus_status[label] = label_result["consensus_reached"]
        termination_reasons[label] = label_result["termination_reason"]
        final_ids_by_label[label] = label_result["final_candidate_ids"]
        recommended_ids_by_label[label] = label_result["recommended_candidate_ids"]
        rounds_by_label[label] = label_result["rounds"]
        unresolved_candidates.extend(label_result["unresolved_candidate_ids"])

    strict_global_decision = GlobalMetaAgent(settings).assemble(registry, final_ids_by_label, label_meta_decisions)
    alternative_ids_by_mode = alternative_candidate_ids_by_mode(
        registry=registry,
        vote_matrices=vote_matrices,
        strict_ids_by_label=strict_global_decision["final_candidate_ids_by_label"],
        recommended_ids_by_label=recommended_ids_by_label,
        calibration_config=calibration_config,
    )
    selected_final_ids_by_label = alternative_ids_by_mode.get(selection_policy, alternative_ids_by_mode["strict"])
    global_decision = GlobalMetaAgent(settings).assemble(registry, selected_final_ids_by_label, label_meta_decisions)
    alternative_spans_by_mode = {
        mode: registry.final_spans_by_label(ids_by_label)
        for mode, ids_by_label in alternative_ids_by_mode.items()
    }
    output = {
        "source_units": [unit.to_dict() for unit in source_units],
        "deterministic_pre_expansion_candidate_ids": [candidate.candidate_id for candidate in deterministic_pre_expansion_candidates],
        "cleaning_log": registry.cleaning_log,
        "label_candidate_registries": registry.registries_json(),
        "registry_expansion_history": registry.registry_expansion_history,
        "resolved_proposals": expansion_manager.resolved_proposals,
        "unresolved_proposals": expansion_manager.unresolved_proposals,
        "doctor_reviews_by_label": doctor_reviews_by_label,
        "review_completeness_by_label": {
            label: all(review.get("review_complete") for review in doctor_reviews_by_label.get(label, []))
            for label in LABELS
        },
        "consistency_metrics_by_label": consistency_by_label,
        "candidate_vote_matrices_by_label": vote_matrices,
        "label_meta_decisions": label_meta_decisions,
        "latest_label_meta_decisions": latest_label_meta_decisions,
        "rounds_by_label": rounds_by_label,
        "label_consensus_status": consensus_status,
        "label_termination_reasons": termination_reasons,
        "strict_global_meta_decision": strict_global_decision,
        "global_meta_decision": global_decision,
        "final_selection_policy": selection_policy,
        "final_candidate_ids_by_label": global_decision["final_candidate_ids_by_label"],
        "recommended_candidate_ids_by_label": recommended_ids_by_label,
        "alternative_candidate_ids_by_mode": alternative_ids_by_mode,
        "alternative_spans_by_mode": alternative_spans_by_mode,
        "final_spans_by_label": global_decision["final_spans_by_label"],
        "final_spans": global_decision["final_spans"],
        "missing_labels": [label for label in LABELS if not global_decision["final_candidate_ids_by_label"].get(label)],
        "unresolved_candidates": sorted(set(unresolved_candidates)),
        "cross_label_conflicts": global_decision["cross_label_conflicts"],
        "overall_consensus_reached": all(consensus_status.get(label, False) for label in LABELS) and not global_decision["cross_label_conflicts"],
    }
    if settings.generate_summaries:
        output["generated_summaries"] = SummaryAgent(llm_client).summarize(
            output["final_spans_by_label"],
            overall_consensus_reached=output["overall_consensus_reached"],
            label_consensus_status=consensus_status,
            final_selection_policy=selection_policy,
        )
    return output


def run_one_label_consultation(
    *,
    label: str,
    source_text: str,
    source_units: Sequence[SourceUnit],
    registry: CandidateRegistry,
    expansion_manager: CandidateExpansionManager,
    settings: PipelineSettings,
    llm_client: LLMClient,
    retriever: GuidelineRetriever,
) -> Dict[str, Any]:
    rounds = []
    last_reviews: list[Dict[str, Any]] = []
    last_metrics: Dict[str, Any] = {}
    last_meta: Dict[str, Any] = {}
    best_strict_ids: list[str] = []
    best_strict_round = 0
    best_reviews: list[Dict[str, Any]] = []
    best_metrics: Dict[str, Any] = {}
    best_meta: Dict[str, Any] = {}
    recommended_ids: list[str] = []
    termination_reason = "maximum_rounds_reached_with_unresolved_decisions"
    newly_appended_ids: list[str] = []
    final_review_extension = False
    round_index = 1

    while round_index <= settings.max_rounds or (
        final_review_extension and round_index == settings.max_rounds + 1
    ):
        review_only_round = round_index > settings.max_rounds
        newly_appended_ids = []
        candidates = registry.candidates_for_label(label)
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        reviews = [
            LabelDoctor(
                expert_name=expert,
                label=label,
                settings=settings,
                llm_client=llm_client,
                retriever=retriever,
            ).review_candidates(source_text, candidates, source_units)
            for expert in DOCTOR_NAMES
        ]
        doctor_expansion_requests = doctor_proposed_expansion_requests(
            reviews,
            label=label,
            limit=settings.max_doctor_expansion_requests_per_label_round,
        )
        metrics = consistency_metrics(reviews, candidate_ids)
        meta = LabelMetaAgent(label, settings, llm_client, retriever).decide(
            source_text=source_text,
            candidates=candidates,
            doctor_reviews=reviews,
            source_units=source_units,
            round_index=round_index,
            allow_expansion=not review_only_round,
        )
        added_candidates: list[Candidate] = []
        expansion_requests = merge_expansion_requests(
            doctor_expansion_requests,
            meta["expansion_requests"],
        )
        if expansion_requests:
            added_candidates = expansion_manager.handle_requests(
                expansion_requests,
                proposed_by_agent=meta["agent_role"],
                generation_round=round_index,
            )
            newly_appended_ids = [candidate.candidate_id for candidate in added_candidates]
        unresolved_expansion_outcomes = [
            outcome
            for outcome in expansion_manager.last_request_outcomes
            if not outcome.get("resolved")
        ]
        effective_consensus = evaluate_label_consensus(
            reviews,
            meta["selected_candidate_ids"],
            unresolved_expansion_outcomes,
        )
        meta_unresolved_ids = list(meta.get("unresolved_candidate_ids", []))
        if meta_unresolved_ids:
            effective_consensus["consensus_reached"] = False
            effective_consensus["unresolved_candidate_ids"] = sorted(
                set(effective_consensus["unresolved_candidate_ids"])
                | set(meta_unresolved_ids)
            )
        strict_ids, boundary_rejections = strict_selected_candidate_ids(
            candidates,
            meta["selected_candidate_ids"],
            metrics["unanimous_keep_candidate_ids"],
        )
        strict_ids, containment_removed = prune_contained_strict_candidates(
            candidates,
            strict_ids,
        )
        proof_missing_ids = missing_consensus_proof_ids(strict_ids, meta)
        if boundary_rejections:
            effective_consensus["consensus_reached"] = False
            effective_consensus["boundary_rejected_selected_ids"] = boundary_rejections
        if proof_missing_ids:
            effective_consensus["consensus_reached"] = False
            effective_consensus["proof_missing_selected_ids"] = proof_missing_ids
            effective_consensus["unresolved_candidate_ids"] = sorted(
                set(effective_consensus["unresolved_candidate_ids"]) | set(proof_missing_ids)
            )
        if better_strict_selection(strict_ids, best_strict_ids):
            best_strict_ids = strict_ids
            best_strict_round = round_index
            best_reviews = reviews
            best_metrics = metrics
            best_meta = meta

        round_record = {
            "round": round_index,
            "candidate_ids_reviewed": candidate_ids,
            "doctor_reviews": reviews,
            "candidate_vote_matrix": metrics["vote_matrix"],
            "consistency_metrics": metrics,
            "label_meta_decision": meta,
            "doctor_expansion_requests": doctor_expansion_requests,
            "label_meta_expansion_requests": meta["expansion_requests"],
            "expansion_requests": expansion_requests,
            "proposal_outcomes": list(expansion_manager.last_request_outcomes),
            "unresolved_proposal_requests": unresolved_expansion_outcomes,
            "new_candidate_ids": newly_appended_ids,
            "new_candidates_require_review": bool(newly_appended_ids),
            "strict_unanimous_selected_ids": strict_ids,
            "boundary_rejected_selected_ids": boundary_rejections,
            "proof_missing_selected_ids": proof_missing_ids,
            "contained_selected_ids_removed": containment_removed,
            "consensus_reached": bool(effective_consensus["consensus_reached"]) and not newly_appended_ids,
        }
        round_record["label_meta_decision"]["consensus"] = effective_consensus
        rounds.append(round_record)
        last_reviews = reviews
        last_metrics = metrics
        last_meta = meta
        recommended_ids = list(meta["selected_candidate_ids"])

        if newly_appended_ids:
            # The new IDs are not accepted until all three doctors review them in
            # the next round as part of the shared label bank.
            if round_index == settings.max_rounds:
                termination_reason = "maximum_rounds_reached_with_unreviewed_new_candidates"
                final_review_extension = True
            round_index += 1
            continue
        if round_record["consensus_reached"]:
            best_strict_ids = strict_ids
            best_strict_round = round_index
            best_reviews = reviews
            best_metrics = metrics
            best_meta = meta
            termination_reason = "strict_unanimous_candidate_consensus"
            break
        if review_only_round or round_index == settings.max_rounds:
            termination_reason = "maximum_rounds_reached_with_unresolved_decisions"
        round_index += 1

    unresolved = []
    if last_meta:
        unresolved.extend(last_meta.get("unresolved_candidate_ids", []))
    if termination_reason == "maximum_rounds_reached_with_unreviewed_new_candidates":
        unresolved.extend(newly_appended_ids)
    unresolved.extend(candidate_id for candidate_id in recommended_ids if candidate_id not in best_strict_ids)

    return {
        "label": label,
        "rounds": rounds,
        "doctor_reviews": best_reviews or last_reviews,
        "consistency_metrics": best_metrics or last_metrics,
        "candidate_vote_matrix": (best_metrics or last_metrics).get("vote_matrix", {}),
        "label_meta_decision": best_meta or last_meta,
        "latest_label_meta_decision": last_meta,
        "consensus_reached": termination_reason == "strict_unanimous_candidate_consensus",
        "termination_reason": termination_reason,
        "final_candidate_ids": best_strict_ids,
        "recommended_candidate_ids": recommended_ids,
        "best_strict_selection_round": best_strict_round,
        "unresolved_candidate_ids": sorted(set(unresolved)),
    }


def doctor_proposed_expansion_requests(
    reviews: Sequence[Mapping[str, Any]],
    *,
    label: str,
    limit: int,
) -> list[Dict[str, Any]]:
    requests: list[Dict[str, Any]] = []
    seen = set()
    for review in reviews:
        for request in review.get("expansion_requests", []):
            cleaned = normalized_expansion_request(request, label)
            if not cleaned:
                continue
            signature = expansion_request_signature(cleaned)
            if signature in seen:
                continue
            seen.add(signature)
            requests.append(cleaned)
            if len(requests) >= max(0, limit):
                return requests
    return requests


def merge_expansion_requests(
    doctor_requests: Sequence[Mapping[str, Any]],
    meta_requests: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    merged: list[Dict[str, Any]] = []
    seen = set()
    for request in list(doctor_requests) + list(meta_requests):
        cleaned = dict(request)
        signature = expansion_request_signature(cleaned)
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(cleaned)
    return merged


def normalized_expansion_request(request: Mapping[str, Any], label: str) -> Dict[str, Any] | None:
    if request.get("label") != label:
        return None
    operation = str(request.get("requested_operation", "")).strip()
    proposed_text = str(request.get("proposed_text", "")).strip()
    if operation != "exact_quote_proposal" or not proposed_text:
        return None
    cleaned = {
        "label": label,
        "requested_operation": "exact_quote_proposal",
        "proposed_text": proposed_text,
        "justification": str(request.get("justification", "doctor proposed exact quote"))[:240],
    }
    sentence_id = str(request.get("sentence_id", "")).strip()
    if sentence_id:
        cleaned["sentence_id"] = sentence_id
    parent_candidate_id = str(request.get("parent_candidate_id", "")).strip()
    if parent_candidate_id:
        cleaned["parent_candidate_id"] = parent_candidate_id
    return cleaned


def expansion_request_signature(request: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        request.get("label"),
        request.get("requested_operation"),
        request.get("parent_candidate_id"),
        request.get("sentence_id"),
        request.get("proposed_text"),
    )


def missing_consensus_proof_ids(
    selected_ids: Sequence[str],
    meta_decision: Mapping[str, Any],
) -> list[str]:
    proof_ids = {str(row.get("candidate_id", "")) for row in meta_decision.get("consensus_proof", [])}
    return [candidate_id for candidate_id in selected_ids if candidate_id not in proof_ids]


def strict_selected_candidate_ids(
    candidates: Sequence[Candidate],
    selected_ids: Sequence[str],
    unanimous_keep_ids: Sequence[str],
) -> tuple[list[str], list[Dict[str, str]]]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    unanimous = set(unanimous_keep_ids)
    accepted = []
    rejected = []
    for candidate_id in selected_ids:
        candidate = by_id.get(candidate_id)
        if not candidate or candidate_id not in unanimous:
            continue
        issue = boundary_quality_issue(candidate.label, candidate.text)
        if issue:
            rejected.append({"candidate_id": candidate_id, "reason": issue})
            continue
        accepted.append(candidate_id)
    return accepted, rejected


def prune_contained_strict_candidates(
    candidates: Sequence[Candidate],
    selected_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    kept = list(selected_ids)
    removed = []
    for left_id in list(selected_ids):
        if left_id not in kept:
            continue
        left = by_id[left_id]
        for right_id in list(selected_ids):
            if left_id == right_id or right_id not in kept:
                continue
            right = by_id[right_id]
            if not (
                left.start <= right.start
                and right.end <= left.end
                and (left.start, left.end) != (right.start, right.end)
            ):
                continue
            preferred = preferred_contained_candidate(left, right)
            rejected = right if preferred.candidate_id == left.candidate_id else left
            if rejected.candidate_id in kept:
                kept.remove(rejected.candidate_id)
                removed.append(rejected.candidate_id)
            if rejected.candidate_id == left_id:
                break
    return kept, sorted(set(removed))


def preferred_contained_candidate(outer: Candidate, inner: Candidate) -> Candidate:
    if outer.label in {"SampleSize", "DesignDescription", "StatisticalAnalysis"}:
        return outer
    if outer.label == "ComparisonGroup":
        outer_words = len(outer.text.split())
        inner_words = len(inner.text.split())
        if inner_words >= 2:
            return inner
        return outer
    outer_issue = boundary_quality_issue(outer.label, outer.text)
    inner_issue = boundary_quality_issue(inner.label, inner.text)
    if outer_issue and not inner_issue:
        return inner
    if inner_issue and not outer_issue:
        return outer
    return inner if len(inner.text) < len(outer.text) else outer


def better_strict_selection(candidate_ids: Sequence[str], current_ids: Sequence[str]) -> bool:
    if len(candidate_ids) != len(current_ids):
        return len(candidate_ids) > len(current_ids)
    return bool(candidate_ids) and tuple(candidate_ids) < tuple(current_ids)


def alternative_candidate_ids_by_mode(
    *,
    registry: CandidateRegistry,
    vote_matrices: Mapping[str, Mapping[str, Mapping[str, str]]],
    strict_ids_by_label: Mapping[str, Sequence[str]],
    recommended_ids_by_label: Mapping[str, Sequence[str]],
    calibration_config: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, list[str]]]:
    modes: Dict[str, Dict[str, list[str]]] = {
        "strict": {label: list(strict_ids_by_label.get(label, [])) for label in LABELS},
        "recommended": {label: list(recommended_ids_by_label.get(label, [])) for label in LABELS},
        "pubmedbert_only": {label: [] for label in LABELS},
        "majority_vote": {label: [] for label in LABELS},
        "pubmedbert_anchor": {label: [] for label in LABELS},
        "pubmedbert_plus_verified": {label: [] for label in LABELS},
        "expert_consensus_verified": {label: [] for label in LABELS},
        "calibrated": {label: [] for label in LABELS},
        "registry_filtered": {label: [] for label in LABELS},
        "registry_all": {label: [] for label in LABELS},
    }
    for label in LABELS:
        candidates = registry.candidates_for_label(label)
        matrix = vote_matrices.get(label, {})
        modes["registry_all"][label] = [candidate.candidate_id for candidate in candidates]
        modes["registry_filtered"][label] = [candidate.candidate_id for candidate in candidates if passes_label_filter(candidate)]
        modes["pubmedbert_only"][label] = [
            candidate.candidate_id
            for candidate in candidates
            if proposed_by_pubmedbert(candidate) and passes_label_filter(candidate)
        ]
        majority_ids = []
        anchor_ids = []
        for candidate in candidates:
            if not passes_label_filter(candidate):
                continue
            votes = list(matrix.get(candidate.candidate_id, {}).values())
            keep_count = sum(vote == "keep" for vote in votes)
            if keep_count >= 2:
                majority_ids.append(candidate.candidate_id)
            if proposed_by_pubmedbert(candidate):
                if keep_count >= 1:
                    anchor_ids.append(candidate.candidate_id)
            elif keep_count >= 2:
                anchor_ids.append(candidate.candidate_id)
        modes["majority_vote"][label], _ = prune_contained_strict_candidates(candidates, majority_ids)
        modes["pubmedbert_anchor"][label], _ = prune_contained_strict_candidates(candidates, anchor_ids)
        modes["pubmedbert_plus_verified"][label] = select_pubmedbert_plus_verified(candidates, matrix)
        modes["expert_consensus_verified"][label] = select_expert_consensus_verified(candidates, matrix)
        if calibration_config is not None:
            label_config = calibration_config.get("labels", {}).get(label, {})
            modes["calibrated"][label] = select_calibrated_candidate_ids(candidates, matrix, label_config)
    return modes


def proposed_by_pubmedbert(candidate: Candidate) -> bool:
    return any(expert.lower() == "pubmedbert" for expert in candidate.proposed_by_experts)
