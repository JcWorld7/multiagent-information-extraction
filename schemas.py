"""Structured-output schemas for vLLM/OpenAI-compatible calls."""
from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from Labelcentered.settings import LABELS

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["keep", "reject"]},
        "issue_type": {
            "type": "string",
            "enum": [
                "wrong_label",
                "boundary_too_short",
                "boundary_too_long",
                "incomplete_phrase",
                "malformed_pdf_text",
                "duplicate",
                "unsupported",
                "missing_span",
                "acceptable",
            ],
        },
        "justification": {"type": "string", "maxLength": 240},
    },
    "required": ["candidate_id", "decision", "issue_type", "justification"],
}

EXPANSION_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "parent_candidate_id": {"type": "string"},
        "requested_operation": {
            "type": "string",
            "enum": [
                "split_by_sentence",
                "split_by_clause",
                "trim_leading_stopword",
                "trim_trailing_incomplete_clause",
                "expand_to_sentence_start",
                "expand_to_sentence_end",
                "generate_number_centered_span",
                "search_missing_label",
                "exact_quote_proposal",
            ],
        },
        "proposed_text": {"type": "string"},
        "sentence_id": {"type": "string"},
        "justification": {"type": "string", "maxLength": 240},
    },
    "required": ["label", "requested_operation", "justification"],
}

DOCTOR_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {"type": "array", "items": DECISION_SCHEMA},
        "expansion_requests": {"type": "array", "items": EXPANSION_REQUEST_SCHEMA},
        "missing_label": {"type": "boolean"},
        "comment": {"type": "string", "maxLength": 240},
    },
    "required": ["decisions", "expansion_requests", "missing_label", "comment"],
}

CONSENSUS_PROOF_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "evidence_sentence_id": {"type": "string"},
        "evidence_quote": {"type": "string", "maxLength": 1000},
        "label_fit_rationale": {"type": "string", "maxLength": 360},
        "boundary_rationale": {"type": "string", "maxLength": 360},
        "why_not_other_labels": {
            "type": "object",
            "additionalProperties": {"type": "string", "maxLength": 240},
        },
        "counterevidence_checked": {"type": "boolean"},
    },
    "required": [
        "candidate_id",
        "evidence_sentence_id",
        "evidence_quote",
        "label_fit_rationale",
        "boundary_rationale",
        "why_not_other_labels",
        "counterevidence_checked",
    ],
}

SOURCE_SEARCH_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "reason": {"type": "string", "maxLength": 300},
        "search_cues": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 8,
        },
        "requested_operation": {
            "type": "string",
            "enum": ["search_missing_label", "verify_consensus", "repair_boundary"],
        },
    },
    "required": ["label", "reason", "search_cues", "requested_operation"],
}

LABEL_META_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_candidate_ids": {"type": "array", "items": {"type": "string"}},
        "rejected_candidate_ids": {"type": "array", "items": {"type": "string"}},
        "expansion_requests": {"type": "array", "items": EXPANSION_REQUEST_SCHEMA},
        "unresolved_candidate_ids": {"type": "array", "items": {"type": "string"}},
        "consensus_proof": {"type": "array", "items": CONSENSUS_PROOF_SCHEMA},
        "source_search_requests": {"type": "array", "items": SOURCE_SEARCH_REQUEST_SCHEMA},
        "rationale": {"type": "string"},
    },
    "required": [
        "selected_candidate_ids",
        "rejected_candidate_ids",
        "expansion_requests",
        "unresolved_candidate_ids",
        "consensus_proof",
        "source_search_requests",
        "rationale",
    ],
}

GLOBAL_META_SCHEMA = {
    "type": "object",
    "properties": {
        "final_candidate_ids_by_label": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string"}},
        },
        "cross_label_conflicts": {"type": "array", "items": {"type": "object"}},
        "unresolved_candidates": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": [
        "final_candidate_ids_by_label",
        "cross_label_conflicts",
        "unresolved_candidates",
        "rationale",
    ],
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summaries_by_label": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "overall_summary": {"type": "string"},
    },
    "required": ["summaries_by_label", "overall_summary"],
}


def constrained_string_items(allowed_values: Sequence[str]) -> dict:
    values = list(dict.fromkeys(allowed_values))
    items = {"type": "string"}
    if values:
        items["enum"] = values
    return items


def doctor_review_schema(candidate_ids: Sequence[str], label: str) -> dict:
    schema = deepcopy(DOCTOR_REVIEW_SCHEMA)
    ids = list(dict.fromkeys(candidate_ids))
    decisions = schema["properties"]["decisions"]
    decisions["items"]["properties"]["candidate_id"] = constrained_string_items(ids)
    decisions["minItems"] = len(ids)
    decisions["maxItems"] = len(ids)
    expansion_requests = schema["properties"]["expansion_requests"]
    expansion_requests["maxItems"] = 5
    expansion_requests["items"]["properties"]["label"] = {
        "type": "string",
        "enum": [label],
    }
    return schema


def label_meta_schema(
    candidate_ids: Sequence[str],
    sentence_ids: Sequence[str],
    label: str,
    allow_expansion: bool = True,
) -> dict:
    schema = deepcopy(LABEL_META_SCHEMA)
    ids = list(dict.fromkeys(candidate_ids))
    candidate_items = constrained_string_items(ids)
    for field_name in (
        "selected_candidate_ids",
        "rejected_candidate_ids",
        "unresolved_candidate_ids",
    ):
        field = schema["properties"][field_name]
        field["items"] = deepcopy(candidate_items)
        field["maxItems"] = len(ids)

    sentence_values = list(dict.fromkeys(sentence_ids))

    proof = schema["properties"]["consensus_proof"]
    proof["maxItems"] = len(ids)
    proof_properties = proof["items"]["properties"]
    proof_properties["candidate_id"] = deepcopy(candidate_items)
    if sentence_values:
        proof_properties["evidence_sentence_id"] = {
            "type": "string",
            "enum": sentence_values,
        }

    source_search = schema["properties"]["source_search_requests"]
    source_search["maxItems"] = 3 if allow_expansion else 0
    source_search["items"]["properties"]["label"] = {"type": "string", "enum": [label]}

    expansion_requests = schema["properties"]["expansion_requests"]
    expansion_requests["maxItems"] = 5 if allow_expansion else 0
    request_properties = expansion_requests["items"]["properties"]
    request_properties["label"] = {"type": "string", "enum": [label]}
    if sentence_values:
        request_properties["sentence_id"] = {
            "type": "string",
            "enum": sentence_values,
        }
    if ids:
        request_properties["parent_candidate_id"] = {
            "type": "string",
            "enum": ids,
        }
    return schema
