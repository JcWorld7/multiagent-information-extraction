"""Calibration helpers for learned candidate selection configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from Labelcentered.selection_policy import (
    candidate_id,
    candidate_label,
    candidate_experts,
    high_quality_rescue,
    passes_label_filter,
    prune_redundant,
    select_expert_consensus_verified,
    select_pubmedbert_plus_verified,
    weak_fragment,
)
from Labelcentered.settings import DOCTOR_NAMES, LABELS


DEFAULT_VOTE_WEIGHTS = {expert: 1.0 for expert in DOCTOR_NAMES}


def load_calibration_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration config must be a JSON object: {config_path}")
    missing = [label for label in LABELS if label not in payload.get("labels", {})]
    if missing:
        raise ValueError(f"Calibration config is missing labels: {missing}")
    return payload


def candidate_ids_by_label_from_config(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, list[str]]:
    output = {label: [] for label in LABELS}
    for label in LABELS:
        label_config = config.get("labels", {}).get(label, {})
        output[label] = select_label_candidate_ids(payload, label, label_config)
    return output


def select_label_candidate_ids(
    payload: Mapping[str, Any],
    label: str,
    label_config: Mapping[str, Any],
) -> list[str]:
    candidates = payload.get("label_candidate_registries", {}).get(label, [])
    vote_matrix = payload.get("candidate_vote_matrices_by_label", {}).get(label, {})
    mode = str(label_config.get("source_mode", "calibrated_rule"))
    if mode == "recommended":
        recommended = payload.get("recommended_candidate_ids_by_label", {}).get(label, [])
        if not recommended:
            recommended = payload.get("alternative_candidate_ids_by_mode", {}).get("recommended", {}).get(label, [])
        return known_candidate_ids(payload, label, recommended)
    if mode == "registry_all":
        return [candidate_id(candidate) for candidate in candidates]
    if mode == "registry_filtered":
        return [candidate_id(candidate) for candidate in candidates if passes_label_filter(candidate)]
    if mode == "expert_consensus_verified":
        return select_expert_consensus_verified(candidates, vote_matrix)
    if mode == "pubmedbert_plus_verified":
        return select_pubmedbert_plus_verified(candidates, vote_matrix)
    return select_calibrated_candidate_ids(candidates, vote_matrix, label_config)


def known_candidate_ids(
    payload: Mapping[str, Any],
    label: str,
    candidate_ids: Sequence[Any],
) -> list[str]:
    known = {
        str(candidate.get("candidate_id"))
        for candidate in payload.get("label_candidate_registries", {}).get(label, [])
    }
    return [str(candidate_id) for candidate_id in candidate_ids if str(candidate_id) in known]


def select_calibrated_candidate_ids(
    candidates: Sequence[Any],
    vote_matrix: Mapping[str, Mapping[str, str]],
    label_config: Mapping[str, Any],
) -> list[str]:
    require_label_filter = bool(label_config.get("require_label_filter", True))
    reject_weak_fragments = bool(label_config.get("reject_weak_fragments", True))
    min_vote_fraction = vote_fraction_threshold(label_config.get("min_vote_fraction", 2 / 3))
    min_expert_support = int(label_config.get("min_expert_support", 2))
    min_confidence = float(label_config.get("min_confidence", 0.0))
    use_high_quality_rescue = bool(label_config.get("use_high_quality_rescue", True))
    vote_weights = normalized_vote_weights(label_config.get("vote_weights", DEFAULT_VOTE_WEIGHTS))

    kept = []
    for candidate in candidates:
        if require_label_filter and not passes_label_filter(candidate):
            continue
        if reject_weak_fragments and weak_fragment(candidate):
            continue
        cid = candidate_id(candidate)
        vote_ok = weighted_keep_fraction(vote_matrix.get(cid, {}), vote_weights) >= min_vote_fraction
        expert_ok = (
            expert_support_count(candidate) >= min_expert_support
            and max_expert_confidence(candidate) >= min_confidence
        )
        rescue_ok = use_high_quality_rescue and high_quality_rescue(candidate, candidates, vote_matrix)
        if vote_ok or expert_ok or rescue_ok:
            kept.append(candidate)
    return [candidate_id(candidate) for candidate in prune_redundant(kept)]


def normalized_vote_weights(raw_weights: Mapping[str, Any]) -> dict[str, float]:
    weights = dict(DEFAULT_VOTE_WEIGHTS)
    for expert in DOCTOR_NAMES:
        try:
            weights[expert] = max(0.0, float(raw_weights.get(expert, weights[expert])))
        except (TypeError, ValueError):
            weights[expert] = DEFAULT_VOTE_WEIGHTS[expert]
    return weights


def weighted_keep_fraction(
    votes_by_role: Mapping[str, str],
    vote_weights: Mapping[str, float],
) -> float:
    total = sum(vote_weights.get(expert, 0.0) for expert in DOCTOR_NAMES)
    if total <= 0:
        return 0.0
    kept = 0.0
    for role, vote in votes_by_role.items():
        if vote != "keep":
            continue
        kept += vote_weights.get(expert_from_doctor_role(role), 0.0)
    return kept / total


def vote_fraction_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 2 / 3
    if abs(threshold - 0.67) < 0.005:
        return 2 / 3
    return threshold


def expert_from_doctor_role(role: str) -> str:
    role_text = str(role)
    for expert in DOCTOR_NAMES:
        if role_text.startswith(f"{expert}_") or role_text == expert:
            return expert
    return role_text.split("_", 1)[0]


def expert_support_count(candidate: Any) -> int:
    return len({expert.lower() for expert in candidate_experts(candidate)})


def max_expert_confidence(candidate: Any) -> float:
    values = confidence_values(candidate)
    return max(values) if values else 0.0


def confidence_values(candidate: Any) -> list[float]:
    if isinstance(candidate, Mapping):
        raw = candidate.get("confidence_by_expert", {})
    else:
        raw = getattr(candidate, "confidence_by_expert", {})
    if not isinstance(raw, Mapping):
        return []
    output = []
    for value in raw.values():
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            continue
    return output


def spans_for_candidate_ids(
    payload: Mapping[str, Any],
    ids_by_label: Mapping[str, Sequence[str]],
) -> dict[str, list[dict[str, Any]]]:
    registry = {
        str(candidate["candidate_id"]): candidate
        for rows in payload.get("label_candidate_registries", {}).values()
        for candidate in rows
    }
    return {
        label: [candidate_to_final_span(registry[cid]) for cid in ids_by_label.get(label, []) if cid in registry]
        for label in LABELS
    }


def candidate_to_final_span(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "label": str(candidate["label"]),
        "text": str(candidate["text"]),
        "start": int(candidate["start"]),
        "end": int(candidate["end"]),
        "confidence_by_expert": dict(candidate.get("confidence_by_expert", {})),
        "threshold_by_expert": dict(candidate.get("threshold_by_expert", {})),
        "proposed_by_experts": list(candidate.get("proposed_by_experts", [])),
        "source_section": str(candidate.get("source_section", "unknown")),
        "generation_origin": str(candidate.get("generation_origin", "unknown")),
    }
