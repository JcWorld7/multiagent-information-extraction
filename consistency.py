"""Review validation, vote matrices, and agreement metrics."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from Labelcentered.settings import DOCTOR_NAMES


def clean_ids(values: Iterable[Any]) -> List[str]:
    out = []
    seen = set()
    for value in values or []:
        candidate_id = str(value).strip()
        if candidate_id and candidate_id not in seen:
            out.append(candidate_id)
            seen.add(candidate_id)
    return out


def validate_review_partition(keep_ids: Iterable[Any], reject_ids: Iterable[Any], assigned_ids: Sequence[str]) -> Dict[str, Any]:
    keep = set(clean_ids(keep_ids))
    reject = set(clean_ids(reject_ids))
    assigned = set(assigned_ids)
    submitted = keep | reject
    unknown = sorted(submitted - assigned)
    overlap = sorted(keep & reject)
    missing = [candidate_id for candidate_id in assigned_ids if candidate_id not in submitted]
    duplicate_decisions = [candidate_id for candidate_id in overlap]
    empty = bool(assigned_ids) and not submitted
    return {
        "valid": not unknown and not overlap and not missing and not empty,
        "missing_ids": missing,
        "unknown_ids": unknown,
        "overlap_ids": overlap,
        "duplicate_decisions": duplicate_decisions,
        "empty_review": empty,
    }


def normalize_review(agent_role: str, response: Mapping[str, Any], assigned_ids: Sequence[str]) -> Dict[str, Any]:
    keep_ids, reject_ids = [], []
    decisions_by_candidate = {}
    for item in response.get("decisions", []):
        candidate_id = str(item.get("candidate_id", "")).strip()
        decision = item.get("decision")
        if decision == "keep":
            keep_ids.append(candidate_id)
        elif decision == "reject":
            reject_ids.append(candidate_id)
    if not keep_ids and not reject_ids:
        keep_ids = clean_ids(response.get("keep_ids", []))
        reject_ids = clean_ids(response.get("reject_ids", []))
    validation = validate_review_partition(keep_ids, reject_ids, assigned_ids)
    assigned = set(assigned_ids)
    if not validation["unknown_ids"] and not validation["overlap_ids"] and not validation["empty_review"]:
        for candidate_id in keep_ids:
            if candidate_id in assigned:
                decisions_by_candidate[candidate_id] = "keep"
        for candidate_id in reject_ids:
            if candidate_id in assigned:
                decisions_by_candidate[candidate_id] = "reject"
    return {
        "agent_role": agent_role,
        "keep_ids": clean_ids(keep_ids),
        "reject_ids": clean_ids(reject_ids),
        "decisions_by_candidate": decisions_by_candidate,
        "expansion_requests": list(response.get("expansion_requests", [])),
        "missing_label": bool(response.get("missing_label", False)),
        "comment": response.get("comment", ""),
        "validation": validation,
        "review_complete": validation["valid"],
    }


def build_vote_matrix(reviews: Sequence[Mapping[str, Any]], candidate_ids: Sequence[str]) -> Dict[str, Dict[str, str]]:
    matrix: Dict[str, Dict[str, str]] = {candidate_id: {} for candidate_id in candidate_ids}
    for review in reviews:
        role = review["agent_role"]
        decisions = review.get("decisions_by_candidate", {})
        for candidate_id in candidate_ids:
            matrix[candidate_id][role] = decisions.get(candidate_id, "missing")
    return matrix


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = 0.0
    for label in ("keep", "reject"):
        pe += (sum(x == label for x in a) / n) * (sum(x == label for x in b) / n)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(rows: Sequence[Sequence[str]]) -> float | None:
    if not rows:
        return None
    n_raters = len(rows[0])
    if n_raters < 2:
        return None
    categories = ("keep", "reject")
    p_bar = 0.0
    totals = {category: 0 for category in categories}
    for row in rows:
        counts = {category: row.count(category) for category in categories}
        for category in categories:
            totals[category] += counts[category]
        p_bar += (sum(c * c for c in counts.values()) - n_raters) / (n_raters * (n_raters - 1))
    p_bar /= len(rows)
    total = len(rows) * n_raters
    p_e = sum((totals[category] / total) ** 2 for category in categories)
    if p_e == 1:
        return 1.0 if p_bar == 1 else 0.0
    return (p_bar - p_e) / (1 - p_e)


def consistency_metrics(reviews: Sequence[Mapping[str, Any]], candidate_ids: Sequence[str]) -> Dict[str, Any]:
    matrix = build_vote_matrix(reviews, candidate_ids)
    roles = [review["agent_role"] for review in reviews]
    reviewed_by_all = [
        candidate_id for candidate_id in candidate_ids
        if all(matrix[candidate_id].get(role) in {"keep", "reject"} for role in roles)
    ]
    rows = [[matrix[candidate_id][role] for role in roles] for candidate_id in reviewed_by_all]
    unanimous_keep = [cid for cid, row in zip(reviewed_by_all, rows) if all(v == "keep" for v in row)]
    unanimous_reject = [cid for cid, row in zip(reviewed_by_all, rows) if all(v == "reject" for v in row)]
    conflicted = [cid for cid, row in zip(reviewed_by_all, rows) if "keep" in row and "reject" in row]
    missing_decision = [cid for cid in candidate_ids if any(matrix[cid].get(role) == "missing" for role in roles)]
    pairwise = {}
    for left, right in combinations(roles, 2):
        ids = [cid for cid in candidate_ids if matrix[cid].get(left) in {"keep", "reject"} and matrix[cid].get(right) in {"keep", "reject"}]
        left_values = [matrix[cid][left] for cid in ids]
        right_values = [matrix[cid][right] for cid in ids]
        agree = sum(x == y for x, y in zip(left_values, right_values))
        left_keep = {cid for cid in ids if matrix[cid][left] == "keep"}
        right_keep = {cid for cid in ids if matrix[cid][right] == "keep"}
        union = left_keep | right_keep
        pairwise[f"{left}__{right}"] = {
            "n": len(ids),
            "raw_agreement": _safe_div(agree, len(ids)),
            "cohen_kappa": cohen_kappa(left_values, right_values),
            "keep_set_jaccard": 1.0 if not union else len(left_keep & right_keep) / len(union),
        }
    return {
        "n_candidates_reviewed_by_all_three": len(reviewed_by_all),
        "pairwise": pairwise,
        "three_way_unanimous_agreement_rate": _safe_div(len(unanimous_keep) + len(unanimous_reject), len(reviewed_by_all)),
        "fleiss_kappa": fleiss_kappa(rows),
        "unanimous_keep_candidate_ids": unanimous_keep,
        "unanimous_reject_candidate_ids": unanimous_reject,
        "conflicted_candidate_ids": conflicted,
        "missing_decision_candidate_ids": missing_decision,
        "vote_matrix": matrix,
        "interpretation": "consistency among three expert-linked prompted roles of the same underlying LLM",
    }


def evaluate_label_consensus(reviews: Sequence[Mapping[str, Any]], selected_ids: Sequence[str], unresolved_expansions: Sequence[Mapping[str, Any]] = ()) -> Dict[str, Any]:
    metrics = consistency_metrics(reviews, selected_ids)
    matrix = metrics["vote_matrix"]
    roles = [review["agent_role"] for review in reviews]
    unknown = sorted({item for review in reviews for item in review.get("validation", {}).get("unknown_ids", [])})
    overlap = sorted({item for review in reviews for item in review.get("validation", {}).get("overlap_ids", [])})
    complete = len(reviews) == 3 and all(review.get("review_complete") for review in reviews)
    all_keep = [
        cid for cid in selected_ids
        if all(matrix.get(cid, {}).get(role) == "keep" for role in roles)
    ]
    majority = [
        cid for cid in selected_ids
        if sum(matrix.get(cid, {}).get(role) == "keep" for role in roles) >= 2
    ]
    unresolved = [
        cid for cid in selected_ids
        if cid not in all_keep
    ]
    has_selection = bool(selected_ids)
    return {
        "review_complete": complete,
        "has_selected_candidates": has_selection,
        "unknown_ids": unknown,
        "overlap_ids": overlap,
        "unanimously_selected": all_keep,
        "majority_selected": majority,
        "unresolved_candidate_ids": unresolved,
        "consensus_reached": (
            complete
            and has_selection
            and not unknown
            and not overlap
            and not unresolved
            and not unresolved_expansions
        ),
        "decision_rule": "unanimous_candidate_level_keep_votes",
    }
