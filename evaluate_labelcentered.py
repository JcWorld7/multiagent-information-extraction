"""Evaluate Labelcentered predictions against Label Studio gold annotations."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.json_input import (
    build_model_input_from_record,
    load_records,
    record_name,
    safe_name,
)
from Labelcentered.selection_policy import passes_label_filter, select_expert_consensus_verified, select_pubmedbert_plus_verified
from Labelcentered.settings import LABELS


@dataclass(frozen=True)
class EvalSpan:
    label: str
    start: int
    end: int
    text: str
    candidate_id: str = ""

    def key(self) -> Tuple[str, int, int]:
        return self.label, self.start, self.end


@dataclass(frozen=True)
class EvalToken:
    start: int
    end: int
    text: str


def safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def prf(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


def prfa(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float | int]:
    return {
        **prf(tp, fp, fn),
        "tn": tn,
        "accuracy": safe_div(tp + tn, tp + fp + fn + tn),
    }


def span_iou(left: EvalSpan, right: EvalSpan) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return safe_div(intersection, union)


def overlapping(left: EvalSpan, right: EvalSpan) -> bool:
    return min(left.end, right.end) > max(left.start, right.start)


def gold_spans_from_record(
    record: Mapping[str, Any],
    source_text: str,
) -> Tuple[List[EvalSpan], List[Dict[str, Any]], List[Dict[str, Any]]]:
    annotations = [
        annotation
        for annotation in record.get("annotations", [])
        if not annotation.get("was_cancelled", False)
    ]
    if not annotations:
        return [], [], []
    result = annotations[0].get("result", [])
    spans = []
    mismatches = []
    repairs = []
    for item in result:
        if item.get("type") != "labels":
            continue
        value = item.get("value", {})
        labels = value.get("labels", [])
        if not labels or labels[0] not in LABELS:
            continue
        label = labels[0]
        start, end = value.get("start"), value.get("end")
        annotation_text = str(value.get("text", ""))
        normalized_annotation = normalize_annotation_text(annotation_text)
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(source_text):
            repaired = realign_gold_span(source_text, normalized_annotation, start, end)
            if repaired:
                repaired_start, repaired_end, repair_method = repaired
                repairs.append({
                    "reason": "invalid_gold_offsets_repaired_from_annotation_text",
                    "label": label,
                    "original_start": start,
                    "original_end": end,
                    "repaired_start": repaired_start,
                    "repaired_end": repaired_end,
                    "repair_method": repair_method,
                    "annotation_text": annotation_text,
                    "repaired_source_text": source_text[repaired_start:repaired_end],
                })
                start, end = repaired_start, repaired_end
            else:
                mismatches.append({"reason": "invalid_gold_offsets", "value": value})
                continue
        reconstructed = source_text[start:end]
        if reconstructed != normalized_annotation:
            repaired = realign_gold_span(source_text, normalized_annotation, start, end)
            if repaired:
                repaired_start, repaired_end, repair_method = repaired
                repairs.append({
                    "reason": "gold_offsets_repaired_from_annotation_text",
                    "label": label,
                    "original_start": start,
                    "original_end": end,
                    "repaired_start": repaired_start,
                    "repaired_end": repaired_end,
                    "repair_method": repair_method,
                    "annotation_text": annotation_text,
                    "original_source_text": reconstructed,
                    "repaired_source_text": source_text[repaired_start:repaired_end],
                })
                start, end = repaired_start, repaired_end
                reconstructed = source_text[start:end]
            elif annotation_equivalent(reconstructed, normalized_annotation):
                repairs.append({
                    "reason": "gold_text_equivalent_after_normalization",
                    "label": label,
                    "start": start,
                    "end": end,
                    "annotation_text": annotation_text,
                    "source_text": reconstructed,
                })
            else:
                mismatches.append({
                    "reason": "annotation_text_differs_from_source_slice",
                    "label": label,
                    "start": start,
                    "end": end,
                    "annotation_text": annotation_text,
                    "source_text": reconstructed,
                })
        spans.append(EvalSpan(label, start, end, reconstructed))
    return spans, mismatches, repairs


def normalize_annotation_text(text: str) -> str:
    return text.replace("\\n", "\n")


def annotation_equivalent(source_slice: str, annotation_text: str) -> bool:
    return comparable_text(source_slice) == comparable_text(annotation_text)


def comparable_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip(" \n\t.;")


def realign_gold_span(
    source_text: str,
    annotation_text: str,
    original_start: Any,
    original_end: Any,
) -> Tuple[int, int, str] | None:
    if not annotation_text.strip():
        return None
    exact = best_exact_match(source_text, annotation_text, original_start)
    if exact:
        return exact[0], exact[1], "exact_annotation_text_match"
    flexible = best_flexible_whitespace_match(source_text, annotation_text, original_start)
    if flexible:
        return flexible[0], flexible[1], "flexible_whitespace_annotation_text_match"
    return None


def best_exact_match(source_text: str, annotation_text: str, original_start: Any) -> Tuple[int, int] | None:
    starts = []
    index = source_text.find(annotation_text)
    while index != -1:
        starts.append(index)
        index = source_text.find(annotation_text, index + 1)
    if not starts:
        return None
    start = min(starts, key=lambda value: offset_distance(value, original_start))
    return start, start + len(annotation_text)


def best_flexible_whitespace_match(source_text: str, annotation_text: str, original_start: Any) -> Tuple[int, int] | None:
    pattern = flexible_whitespace_pattern(annotation_text)
    if not pattern:
        return None
    try:
        regex = re.compile(pattern, flags=re.DOTALL)
    except re.error:
        return None
    matches = list(regex.finditer(source_text))
    if not matches:
        return None
    match = min(matches, key=lambda item: offset_distance(item.start(), original_start))
    return match.start(), match.end()


def flexible_whitespace_pattern(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    parts = []
    last_end = 0
    for match in re.finditer(r"\s+", stripped):
        literal = stripped[last_end:match.start()]
        if literal:
            parts.append(re.escape(literal))
        parts.append(r"\s+")
        last_end = match.end()
    tail = stripped[last_end:]
    if tail:
        parts.append(re.escape(tail))
    return "".join(parts)


def offset_distance(value: int, original_start: Any) -> int:
    return abs(value - original_start) if isinstance(original_start, int) else 0


def prediction_spans(payload: Mapping[str, Any], field: str = "final_spans_by_label") -> List[EvalSpan]:
    rows = []
    for label in LABELS:
        for span in payload.get(field, {}).get(label, []):
            rows.append(EvalSpan(
                label=label,
                start=int(span["start"]),
                end=int(span["end"]),
                text=str(span["text"]),
                candidate_id=str(span.get("candidate_id", "")),
            ))
    return rows


def recommended_spans(payload: Mapping[str, Any]) -> List[EvalSpan]:
    registry = {
        candidate["candidate_id"]: candidate
        for candidates in payload.get("label_candidate_registries", {}).values()
        for candidate in candidates
    }
    rows = []
    for label in LABELS:
        for candidate_id in payload.get("recommended_candidate_ids_by_label", {}).get(label, []):
            candidate = registry.get(candidate_id)
            if candidate:
                rows.append(EvalSpan(
                    label=label,
                    start=int(candidate["start"]),
                    end=int(candidate["end"]),
                    text=str(candidate["text"]),
                    candidate_id=candidate_id,
                ))
    return rows


def registry_spans(payload: Mapping[str, Any]) -> List[EvalSpan]:
    return [
        EvalSpan(
            label=label,
            start=int(candidate["start"]),
            end=int(candidate["end"]),
            text=str(candidate["text"]),
            candidate_id=str(candidate["candidate_id"]),
        )
        for label in LABELS
        for candidate in payload.get("label_candidate_registries", {}).get(label, [])
    ]

SELECTION_ABLATION_MODES = [
    "final",
    "recommended",
    "pubmedbert_only",
    "majority_vote",
    "pubmedbert_anchor",
    "pubmedbert_plus_verified",
    "expert_consensus_verified",
    "registry_filtered",
    "registry_all",
]
DETERMINISTIC_SELECTION_MODES = {
    "pubmedbert_only",
    "majority_vote",
    "pubmedbert_anchor",
    "pubmedbert_plus_verified",
    "expert_consensus_verified",
}


def selection_spans(payload: Mapping[str, Any], mode: str) -> List[EvalSpan]:
    if mode == "final":
        return prediction_spans(payload)
    if mode == "recommended":
        return recommended_spans(payload)
    if mode == "registry_all":
        return registry_spans(payload)
    if mode == "registry_filtered":
        return [span for span in registry_spans(payload) if passes_label_filter(span_to_candidate_dict(span))]
    if mode in DETERMINISTIC_SELECTION_MODES:
        return spans_for_candidate_ids(payload, selection_candidate_ids(payload, mode))
    embedded = payload.get("alternative_spans_by_mode", {}).get(mode)
    if embedded:
        return spans_from_by_label(embedded)
    return spans_for_candidate_ids(payload, selection_candidate_ids(payload, mode))


def spans_from_by_label(by_label: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[EvalSpan]:
    rows = []
    for label in LABELS:
        for span in by_label.get(label, []):
            rows.append(EvalSpan(
                label=label,
                start=int(span["start"]),
                end=int(span["end"]),
                text=str(span["text"]),
                candidate_id=str(span.get("candidate_id", "")),
            ))
    return rows


def spans_for_candidate_ids(payload: Mapping[str, Any], ids_by_label: Mapping[str, Sequence[str]]) -> List[EvalSpan]:
    registry = {
        candidate["candidate_id"]: candidate
        for candidates in payload.get("label_candidate_registries", {}).values()
        for candidate in candidates
    }
    rows = []
    for label in LABELS:
        for candidate_id in ids_by_label.get(label, []):
            candidate = registry.get(candidate_id)
            if candidate:
                rows.append(EvalSpan(
                    label=label,
                    start=int(candidate["start"]),
                    end=int(candidate["end"]),
                    text=str(candidate["text"]),
                    candidate_id=str(candidate_id),
                ))
    return rows



def span_to_candidate_dict(span: EvalSpan) -> Dict[str, Any]:
    return {
        "candidate_id": span.candidate_id,
        "label": span.label,
        "text": span.text,
        "start": span.start,
        "end": span.end,
        "proposed_by_experts": [],
    }


def selection_candidate_ids(payload: Mapping[str, Any], mode: str) -> Dict[str, List[str]]:
    embedded = payload.get("alternative_candidate_ids_by_mode", {}).get(mode)
    if embedded and mode not in DETERMINISTIC_SELECTION_MODES:
        return {label: list(embedded.get(label, [])) for label in LABELS}
    if mode == "pubmedbert_only":
        return pubmedbert_candidate_ids(payload)
    if mode == "majority_vote":
        return vote_based_candidate_ids(payload, pubmedbert_anchor=False)
    if mode == "pubmedbert_anchor":
        return vote_based_candidate_ids(payload, pubmedbert_anchor=True)
    if mode == "pubmedbert_plus_verified":
        return pubmedbert_plus_verified_candidate_ids(payload)
    if mode == "expert_consensus_verified":
        return expert_consensus_verified_candidate_ids(payload)
    return {label: [] for label in LABELS}


def expert_consensus_verified_candidate_ids(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    output = {label: [] for label in LABELS}
    for label in LABELS:
        candidates = payload.get("label_candidate_registries", {}).get(label, [])
        matrix = payload.get("candidate_vote_matrices_by_label", {}).get(label, {})
        output[label] = select_expert_consensus_verified(candidates, matrix)
    return output



def pubmedbert_plus_verified_candidate_ids(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    output = {label: [] for label in LABELS}
    for label in LABELS:
        candidates = payload.get("label_candidate_registries", {}).get(label, [])
        matrix = payload.get("candidate_vote_matrices_by_label", {}).get(label, {})
        output[label] = select_pubmedbert_plus_verified(candidates, matrix)
    return output


def pubmedbert_candidate_ids(payload: Mapping[str, Any]) -> Dict[str, List[str]]:
    output = {label: [] for label in LABELS}
    for label in LABELS:
        for candidate in payload.get("label_candidate_registries", {}).get(label, []):
            experts = {str(expert).lower() for expert in candidate.get("proposed_by_experts", [])}
            if "pubmedbert" in experts and passes_label_filter(candidate):
                output[label].append(str(candidate["candidate_id"]))
    return output


def vote_based_candidate_ids(payload: Mapping[str, Any], pubmedbert_anchor: bool) -> Dict[str, List[str]]:
    output = {label: [] for label in LABELS}
    pubmed_ids = pubmedbert_candidate_ids(payload)
    for label in LABELS:
        pubmed = set(pubmed_ids[label])
        matrix = payload.get("candidate_vote_matrices_by_label", {}).get(label, {})
        ids = []
        for candidate in payload.get("label_candidate_registries", {}).get(label, []):
            candidate_id = str(candidate["candidate_id"])
            votes = list(matrix.get(candidate_id, {}).values())
            keep_count = sum(vote == "keep" for vote in votes)
            if not passes_label_filter(candidate):
                continue
            if pubmedbert_anchor and candidate_id in pubmed:
                if keep_count >= 1:
                    ids.append(candidate_id)
            elif keep_count >= 2:
                ids.append(candidate_id)
        output[label] = ids
    return output


def exact_counts(gold: Sequence[EvalSpan], predicted: Sequence[EvalSpan]) -> Dict[str, int]:
    gold_keys = {span.key() for span in gold}
    predicted_keys = {span.key() for span in predicted}
    return {
        "tp": len(gold_keys & predicted_keys),
        "fp": len(predicted_keys - gold_keys),
        "fn": len(gold_keys - predicted_keys),
    }


def greedy_overlap_matches(
    gold: Sequence[EvalSpan],
    predicted: Sequence[EvalSpan],
    threshold: float,
    require_same_label: bool = True,
) -> List[Tuple[int, int, float]]:
    choices = []
    for gold_index, gold_span in enumerate(gold):
        for predicted_index, predicted_span in enumerate(predicted):
            if require_same_label and gold_span.label != predicted_span.label:
                continue
            score = span_iou(gold_span, predicted_span)
            if score >= threshold:
                choices.append((score, gold_index, predicted_index))
    choices.sort(reverse=True)
    used_gold, used_predicted, matches = set(), set(), []
    for score, gold_index, predicted_index in choices:
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        matches.append((gold_index, predicted_index, score))
    return matches


def per_label_span_metrics(
    documents: Sequence[Tuple[Sequence[EvalSpan], Sequence[EvalSpan]]],
    threshold: float | None,
) -> Dict[str, Any]:
    output = {}
    for label in LABELS:
        tp = fp = fn = 0
        for gold, predicted in documents:
            gold_label = [span for span in gold if span.label == label]
            predicted_label = [span for span in predicted if span.label == label]
            if threshold is None:
                counts = exact_counts(gold_label, predicted_label)
                matched = counts["tp"]
            else:
                matched = len(greedy_overlap_matches(gold_label, predicted_label, threshold))
            tp += matched
            fp += len(predicted_label) - matched
            fn += len(gold_label) - matched
        output[label] = prf(tp, fp, fn)
    output["micro"] = prf(
        sum(output[label]["tp"] for label in LABELS),
        sum(output[label]["fp"] for label in LABELS),
        sum(output[label]["fn"] for label in LABELS),
    )
    output["macro"] = {
        metric: sum(float(output[label][metric]) for label in LABELS) / len(LABELS)
        for metric in ("precision", "recall", "f1")
    }
    return output


def character_metrics(
    documents: Sequence[Tuple[int, Sequence[EvalSpan], Sequence[EvalSpan]]],
) -> Dict[str, Any]:
    output = {}
    for label in LABELS:
        tp = fp = fn = 0
        for text_length, gold, predicted in documents:
            gold_chars = covered_characters(gold, label, text_length)
            predicted_chars = covered_characters(predicted, label, text_length)
            tp += len(gold_chars & predicted_chars)
            fp += len(predicted_chars - gold_chars)
            fn += len(gold_chars - predicted_chars)
        output[label] = prf(tp, fp, fn)
    output["micro"] = prf(
        sum(output[label]["tp"] for label in LABELS),
        sum(output[label]["fp"] for label in LABELS),
        sum(output[label]["fn"] for label in LABELS),
    )
    output["macro"] = {
        metric: sum(float(output[label][metric]) for label in LABELS) / len(LABELS)
        for metric in ("precision", "recall", "f1")
    }
    return output


TOKEN_PATTERN = re.compile(r"\w+(?:[-\u2019']\w+)*|[^\w\s]", flags=re.UNICODE)


def tokenize_with_offsets(source_text: str) -> List[EvalToken]:
    return [
        EvalToken(match.start(), match.end(), match.group(0))
        for match in TOKEN_PATTERN.finditer(source_text)
    ]


def token_indices_for_label(
    tokens: Sequence[EvalToken],
    spans: Sequence[EvalSpan],
    label: str,
) -> set[int]:
    label_spans = [span for span in spans if span.label == label]
    return {
        index
        for index, token in enumerate(tokens)
        if any(
            min(token.end, span.end) > max(token.start, span.start)
            for span in label_spans
        )
    }


def token_level_metrics(
    documents: Sequence[
        Tuple[Sequence[EvalToken], Sequence[EvalSpan], Sequence[EvalSpan]]
    ],
) -> Dict[str, Any]:
    output = {}
    for label in LABELS:
        tp = fp = fn = tn = 0
        for tokens, gold, predicted in documents:
            gold_tokens = token_indices_for_label(tokens, gold, label)
            predicted_tokens = token_indices_for_label(tokens, predicted, label)
            tp += len(gold_tokens & predicted_tokens)
            fp += len(predicted_tokens - gold_tokens)
            fn += len(gold_tokens - predicted_tokens)
            tn += len(tokens) - len(gold_tokens | predicted_tokens)
        output[label] = prfa(tp, fp, fn, tn)
    output["micro"] = prfa(
        sum(output[label]["tp"] for label in LABELS),
        sum(output[label]["fp"] for label in LABELS),
        sum(output[label]["fn"] for label in LABELS),
        sum(output[label]["tn"] for label in LABELS),
    )
    output["macro"] = {
        metric: sum(float(output[label][metric]) for label in LABELS) / len(LABELS)
        for metric in ("precision", "recall", "f1", "accuracy")
    }
    return output


def token_assignment_accuracy(
    documents: Sequence[
        Tuple[Sequence[EvalToken], Sequence[EvalSpan], Sequence[EvalSpan]]
    ],
) -> Dict[str, float | int]:
    correct = total = 0
    labeled_correct = labeled_total = 0
    for tokens, gold, predicted in documents:
        gold_assignments = token_label_assignments(tokens, gold)
        predicted_assignments = token_label_assignments(tokens, predicted)
        for gold_label, predicted_label in zip(gold_assignments, predicted_assignments):
            is_correct = gold_label == predicted_label
            correct += int(is_correct)
            total += 1
            if gold_label != "NONE" or predicted_label != "NONE":
                labeled_correct += int(is_correct)
                labeled_total += 1
    return {
        "correct_tokens": correct,
        "total_tokens": total,
        "accuracy": safe_div(correct, total),
        "labeled_correct_tokens": labeled_correct,
        "labeled_total_tokens": labeled_total,
        "labeled_accuracy": safe_div(labeled_correct, labeled_total),
    }


def token_label_confusion(
    documents: Sequence[
        Tuple[Sequence[EvalToken], Sequence[EvalSpan], Sequence[EvalSpan]]
    ],
) -> Dict[str, Dict[str, int]]:
    categories = LABELS + ["NONE"]
    matrix = {gold: {predicted: 0 for predicted in categories} for gold in categories}
    for tokens, gold, predicted in documents:
        gold_assignments = token_label_assignments(tokens, gold)
        predicted_assignments = token_label_assignments(tokens, predicted)
        for gold_label, predicted_label in zip(gold_assignments, predicted_assignments):
            matrix[gold_label][predicted_label] += 1
    return matrix


def token_label_assignments(
    tokens: Sequence[EvalToken],
    spans: Sequence[EvalSpan],
) -> List[str]:
    priority = {label: index for index, label in enumerate(LABELS)}
    assignments = []
    for token in tokens:
        labels = {
            span.label
            for span in spans
            if min(token.end, span.end) > max(token.start, span.start)
        }
        assignments.append(
            max(labels, key=lambda label: priority[label])
            if labels
            else "NONE"
        )
    return assignments


def registry_token_recall(
    documents: Sequence[
        Tuple[Sequence[EvalToken], Sequence[EvalSpan], Sequence[EvalSpan]]
    ],
) -> Dict[str, Any]:
    output = {}
    for label in LABELS:
        gold_total = matched = 0
        for tokens, gold, candidates in documents:
            gold_tokens = token_indices_for_label(tokens, gold, label)
            candidate_tokens = token_indices_for_label(tokens, candidates, label)
            gold_total += len(gold_tokens)
            matched += len(gold_tokens & candidate_tokens)
        output[label] = {
            "gold_tokens": gold_total,
            "matched_gold_tokens": matched,
            "recall": safe_div(matched, gold_total),
        }
    output["micro"] = {
        "gold_tokens": sum(output[label]["gold_tokens"] for label in LABELS),
        "matched_gold_tokens": sum(
            output[label]["matched_gold_tokens"] for label in LABELS
        ),
    }
    output["micro"]["recall"] = safe_div(
        output["micro"]["matched_gold_tokens"],
        output["micro"]["gold_tokens"],
    )
    output["macro"] = {
        "recall": sum(output[label]["recall"] for label in LABELS) / len(LABELS)
    }
    return output


def covered_characters(spans: Sequence[EvalSpan], label: str, text_length: int) -> set[int]:
    covered = set()
    for span in spans:
        if span.label == label:
            covered.update(range(max(0, span.start), min(text_length, span.end)))
    return covered


def presence_metrics(
    documents: Sequence[Tuple[Sequence[EvalSpan], Sequence[EvalSpan]]],
) -> Dict[str, Any]:
    output = {}
    for label in LABELS:
        tp = fp = fn = tn = 0
        for gold, predicted in documents:
            gold_present = any(span.label == label for span in gold)
            predicted_present = any(span.label == label for span in predicted)
            if gold_present and predicted_present:
                tp += 1
            elif predicted_present:
                fp += 1
            elif gold_present:
                fn += 1
            else:
                tn += 1
        output[label] = {**prf(tp, fp, fn), "tn": tn}
    return output


def boundary_metrics(
    documents: Sequence[Tuple[Sequence[EvalSpan], Sequence[EvalSpan]]],
) -> Dict[str, Any]:
    by_label: Dict[str, List[Tuple[EvalSpan, EvalSpan]]] = {label: [] for label in LABELS}
    for gold, predicted in documents:
        for gold_index, predicted_index, _ in greedy_overlap_matches(gold, predicted, 0.01):
            gold_span, predicted_span = gold[gold_index], predicted[predicted_index]
            by_label[gold_span.label].append((gold_span, predicted_span))
    output = {}
    for label, pairs in by_label.items():
        start_errors = [abs(pred.start - gold.start) for gold, pred in pairs]
        end_errors = [abs(pred.end - gold.end) for gold, pred in pairs]
        output[label] = {
            "matched_spans": len(pairs),
            "exact_boundary_accuracy": safe_div(
                sum(gold.start == pred.start and gold.end == pred.end for gold, pred in pairs),
                len(pairs),
            ),
            "mean_absolute_start_error": safe_div(sum(start_errors), len(start_errors)),
            "mean_absolute_end_error": safe_div(sum(end_errors), len(end_errors)),
            "overextended": sum(pred.start < gold.start or pred.end > gold.end for gold, pred in pairs),
            "underextended": sum(pred.start > gold.start or pred.end < gold.end for gold, pred in pairs),
        }
    return output


def matched_label_confusion(
    documents: Sequence[Tuple[Sequence[EvalSpan], Sequence[EvalSpan]]],
) -> Dict[str, Dict[str, int]]:
    matrix = {label: {predicted: 0 for predicted in LABELS + ["NONE"]} for label in LABELS}
    for gold, predicted in documents:
        matches = greedy_overlap_matches(gold, predicted, 0.01, require_same_label=False)
        matched_gold = set()
        for gold_index, predicted_index, _ in matches:
            gold_span, predicted_span = gold[gold_index], predicted[predicted_index]
            matrix[gold_span.label][predicted_span.label] += 1
            matched_gold.add(gold_index)
        for index, gold_span in enumerate(gold):
            if index not in matched_gold:
                matrix[gold_span.label]["NONE"] += 1
    return matrix


def character_label_confusion(
    documents: Sequence[Tuple[int, Sequence[EvalSpan], Sequence[EvalSpan]]],
) -> Dict[str, Dict[str, int]]:
    categories = LABELS + ["NONE"]
    matrix = {gold: {predicted: 0 for predicted in categories} for gold in categories}
    for text_length, gold, predicted in documents:
        gold_labels = character_label_assignments(gold, text_length)
        predicted_labels = character_label_assignments(predicted, text_length)
        for gold_label, predicted_label in zip(gold_labels, predicted_labels):
            matrix[gold_label][predicted_label] += 1
    return matrix


def character_label_assignments(spans: Sequence[EvalSpan], text_length: int) -> List[str]:
    assignments = ["NONE"] * text_length
    priority = {label: index for index, label in enumerate(LABELS)}
    for span in sorted(spans, key=lambda item: priority[item.label], reverse=True):
        for index in range(max(0, span.start), min(text_length, span.end)):
            assignments[index] = span.label
    return assignments


def oracle_recall(
    documents: Sequence[Tuple[Sequence[EvalSpan], Sequence[EvalSpan]]],
    threshold: float | None,
) -> Dict[str, Any]:
    result = {}
    for label in LABELS:
        total = matched = 0
        for gold, candidates in documents:
            gold_label = [span for span in gold if span.label == label]
            candidate_label = [span for span in candidates if span.label == label]
            total += len(gold_label)
            if threshold is None:
                candidate_keys = {span.key() for span in candidate_label}
                matched += sum(span.key() in candidate_keys for span in gold_label)
            else:
                matched += len(greedy_overlap_matches(gold_label, candidate_label, threshold))
        result[label] = {"gold_spans": total, "matched": matched, "recall": safe_div(matched, total)}
    result["micro"] = {
        "gold_spans": sum(result[label]["gold_spans"] for label in LABELS),
        "matched": sum(result[label]["matched"] for label in LABELS),
    }
    result["micro"]["recall"] = safe_div(result["micro"]["matched"], result["micro"]["gold_spans"])
    return result


def load_prediction_payloads(paths: Sequence[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for index, item in enumerate(payload):
                if isinstance(item, dict):
                    payloads.append((path.with_name(f"{path.name}#{index}"), item))
        elif isinstance(payload, dict):
            payloads.append((path, payload))
    return payloads


def prediction_paths(explicit: Sequence[str], prediction_dir: str | None) -> List[Path]:
    paths = [Path(value).expanduser().resolve() for value in explicit]
    if prediction_dir:
        directory = Path(prediction_dir).expanduser().resolve()
        paths.extend(sorted(directory.glob("*.labelcentered.json")))
    unique = []
    seen = set()
    for path in paths:
        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))
    return unique


def alignment_keys_for_gold(record: Mapping[str, Any], index: int) -> set[str]:
    data = record.get("data", record)
    file_name = str(data.get("file", ""))
    return {
        str(index),
        record_name(dict(record), index),
        safe_name(file_name),
        safe_name(Path(file_name).stem),
    }


def alignment_keys_for_prediction(payload: Mapping[str, Any]) -> set[str]:
    values = [payload.get("record_index"), payload.get("record_name")]
    if payload.get("input_type") != "json":
        values.extend([
            payload.get("source_file"),
            Path(str(payload.get("source_file", ""))).stem,
        ])
    return {safe_name(str(value)) for value in values if value not in (None, "")} | {
        str(payload.get("record_index"))
        for _ in [0]
        if payload.get("record_index") is not None
    }


def align_records(
    gold_records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Tuple[Path, Mapping[str, Any]]],
) -> Tuple[List[Tuple[int, Mapping[str, Any], Path, Mapping[str, Any]]], List[str]]:
    aligned = []
    used_predictions = set()
    warnings = []
    for index, record in enumerate(gold_records):
        gold_keys = alignment_keys_for_gold(record, index)
        matches = [
            (prediction_index, path, payload)
            for prediction_index, (path, payload) in enumerate(predictions)
            if prediction_index not in used_predictions
            and gold_keys & alignment_keys_for_prediction(payload)
        ]
        if len(matches) == 1:
            prediction_index, path, payload = matches[0]
            used_predictions.add(prediction_index)
            aligned.append((index, record, path, payload))
        elif not matches:
            warnings.append(f"No prediction matched gold record {index} ({record_name(dict(record), index)})")
        else:
            warnings.append(f"Multiple predictions matched gold record {index}: {[str(item[1]) for item in matches]}")
    for prediction_index, (path, _) in enumerate(predictions):
        if prediction_index not in used_predictions:
            warnings.append(f"Prediction was not aligned: {path}")
    return aligned, warnings


def evaluate(
    gold_path: Path,
    prediction_files: Sequence[Path],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
) -> Dict[str, Any]:
    gold_records = load_records(gold_path)
    predictions = load_prediction_payloads(prediction_files)
    aligned, warnings = align_records(gold_records, predictions)
    final_documents = []
    recommended_documents = []
    registry_documents = []
    character_documents = []
    final_token_documents = []
    recommended_token_documents = []
    registry_token_documents = []
    ablation_documents = {mode: [] for mode in SELECTION_ABLATION_MODES}
    ablation_token_documents = {mode: [] for mode in SELECTION_ABLATION_MODES}
    annotation_mismatches = []
    annotation_repairs = []
    document_rows = []
    consensus_labels = 0
    total_labels = 0

    for gold_index, record, prediction_path, payload in aligned:
        source_text = build_model_input_from_record(dict(record))
        gold_spans, mismatches, repairs = gold_spans_from_record(record, source_text)
        final_spans = prediction_spans(payload)
        recommended = recommended_spans(payload)
        registry = registry_spans(payload)
        tokens = tokenize_with_offsets(source_text)
        final_documents.append((gold_spans, final_spans))
        recommended_documents.append((gold_spans, recommended))
        registry_documents.append((gold_spans, registry))
        character_documents.append((len(source_text), gold_spans, final_spans))
        final_token_documents.append((tokens, gold_spans, final_spans))
        recommended_token_documents.append((tokens, gold_spans, recommended))
        registry_token_documents.append((tokens, gold_spans, registry))
        for mode in SELECTION_ABLATION_MODES:
            mode_spans = selection_spans(payload, mode)
            ablation_documents[mode].append((gold_spans, mode_spans))
            ablation_token_documents[mode].append((tokens, gold_spans, mode_spans))
        annotation_mismatches.extend([
            {"record_index": gold_index, **mismatch}
            for mismatch in mismatches
        ])
        annotation_repairs.extend([
            {"record_index": gold_index, **repair}
            for repair in repairs
        ])
        statuses = payload.get("label_consensus_status", {})
        consensus_labels += sum(statuses.get(label) is True for label in LABELS)
        total_labels += len(LABELS)
        document_rows.append({
            "record_index": gold_index,
            "record_name": record_name(dict(record), gold_index),
            "prediction_path": str(prediction_path),
            "gold_span_count": len(gold_spans),
            "final_span_count": len(final_spans),
            "recommended_span_count": len(recommended),
            "overall_consensus_reached": payload.get("overall_consensus_reached"),
            "label_consensus_status": statuses,
            "exact": prf(**exact_counts(gold_spans, final_spans)),
        })

    exact = per_label_span_metrics(final_documents, None)
    relaxed = {
        str(threshold): per_label_span_metrics(final_documents, threshold)
        for threshold in iou_thresholds
    }
    recommended_exact = per_label_span_metrics(recommended_documents, None)
    final_token_metrics = token_level_metrics(final_token_documents)
    recommended_token_metrics = token_level_metrics(recommended_token_documents)
    selection_ablation_metrics = {
        mode: {
            "exact_span_metrics": per_label_span_metrics(ablation_documents[mode], None),
            "token_level_metrics": token_level_metrics(ablation_token_documents[mode]),
            "token_assignment_accuracy": token_assignment_accuracy(ablation_token_documents[mode]),
        }
        for mode in SELECTION_ABLATION_MODES
    }
    report = {
        "gold_path": str(gold_path),
        "prediction_files": [str(path) for path in prediction_files],
        "gold_record_count": len(gold_records),
        "prediction_payload_count": len(predictions),
        "aligned_record_count": len(aligned),
        "alignment_warnings": warnings,
        "annotation_text_mismatch_count": len(annotation_mismatches),
        "annotation_text_mismatches": annotation_mismatches,
        "gold_annotation_repair_count": len(annotation_repairs),
        "gold_annotation_repairs": annotation_repairs,
        "documents": document_rows,
        "final_exact_span_metrics": exact,
        "final_relaxed_span_metrics": relaxed,
        "recommended_exact_span_metrics": recommended_exact,
        "final_token_level_metrics": final_token_metrics,
        "final_token_assignment_accuracy": token_assignment_accuracy(final_token_documents),
        "recommended_token_level_metrics": recommended_token_metrics,
        "selection_ablation_metrics": selection_ablation_metrics,
        "registry_oracle_token_recall": registry_token_recall(
            registry_token_documents
        ),
        "token_label_confusion": token_label_confusion(final_token_documents),
        "character_level_metrics": character_metrics(character_documents),
        "character_label_confusion": character_label_confusion(character_documents),
        "document_label_presence_metrics": presence_metrics(final_documents),
        "boundary_metrics": boundary_metrics(final_documents),
        "matched_span_label_confusion": matched_label_confusion(final_documents),
        "registry_oracle_exact_recall": oracle_recall(registry_documents, None),
        "registry_oracle_iou_0.5_recall": oracle_recall(registry_documents, 0.5),
        "consensus_coverage": {
            "labels_with_consensus": consensus_labels,
            "total_document_labels": total_labels,
            "rate": safe_div(consensus_labels, total_labels),
        },
    }
    return report


def readable_report(report: Mapping[str, Any]) -> str:
    lines = [
        "LABELCENTERED EVALUATION",
        f"Gold records: {report['gold_record_count']}",
        f"Prediction payloads: {report['prediction_payload_count']}",
        f"Aligned records: {report['aligned_record_count']}",
        f"Gold annotation text/source mismatches: {report['annotation_text_mismatch_count']}",
        f"Gold annotation offset/text repairs: {report.get('gold_annotation_repair_count', 0)}",
        "",
        "FINAL EXACT SPAN METRICS",
    ]
    exact = report["final_exact_span_metrics"]
    for label in LABELS:
        row = exact[label]
        lines.append(
            f"{label:22} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )
    lines.extend([
        "",
        f"Micro F1: {exact['micro']['f1']:.4f}",
        f"Macro F1: {exact['macro']['f1']:.4f}",
    ])
    for threshold, metrics in report["final_relaxed_span_metrics"].items():
        lines.extend([
            "",
            f"RELAXED SPAN METRICS (IoU >= {threshold})",
            f"Micro F1: {metrics['micro']['f1']:.4f}",
            f"Macro F1: {metrics['macro']['f1']:.4f}",
        ])
    character = report["character_level_metrics"]
    token = report["final_token_level_metrics"]
    recommended_token = report["recommended_token_level_metrics"]
    registry_token = report["registry_oracle_token_recall"]
    oracle_exact = report["registry_oracle_exact_recall"]["micro"]
    oracle_overlap = report["registry_oracle_iou_0.5_recall"]["micro"]
    coverage = report["consensus_coverage"]
    lines.extend([
        "",
        "CHARACTER-LEVEL METRICS",
        f"Micro F1: {character['micro']['f1']:.4f}",
        f"Macro F1: {character['macro']['f1']:.4f}",
        "",
        "TOKEN-LEVEL METRICS (STRICT FINAL SPANS)",
    ])
    for label in LABELS:
        row = token[label]
        lines.append(
            f"{label:22} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} Acc={row['accuracy']:.4f} "
            f"TP={row['tp']} FP={row['fp']} FN={row['fn']} TN={row['tn']}"
        )
    assignment_accuracy = report["final_token_assignment_accuracy"]
    lines.extend([
        f"Micro F1: {token['micro']['f1']:.4f}",
        f"Micro accuracy (one-vs-rest): {token['micro']['accuracy']:.4f}",
        f"Macro F1: {token['macro']['f1']:.4f}",
        f"Macro accuracy (one-vs-rest): {token['macro']['accuracy']:.4f}",
        f"Overall token assignment accuracy: {assignment_accuracy['accuracy']:.4f}",
        f"Labeled-token assignment accuracy: {assignment_accuracy['labeled_accuracy']:.4f}",
        "",
        "TOKEN-LEVEL DIAGNOSTICS",
        f"Recommended spans micro F1: {recommended_token['micro']['f1']:.4f}",
        f"Recommended spans macro F1: {recommended_token['macro']['f1']:.4f}",
        f"Registry oracle token micro recall: {registry_token['micro']['recall']:.4f}",
        f"Registry oracle token macro recall: {registry_token['macro']['recall']:.4f}",
        "",
        "SELECTION ABLATION TOKEN MICRO F1",
    ])
    for mode, metrics in report.get("selection_ablation_metrics", {}).items():
        micro = metrics["token_level_metrics"]["micro"]
        lines.append(
            f"{mode:22} P={micro['precision']:.4f} R={micro['recall']:.4f} "
            f"F1={micro['f1']:.4f} Acc={micro['accuracy']:.4f}"
        )
    lines.extend([
        "",
        "ARCHITECTURE DIAGNOSTICS",
        f"Registry oracle exact recall: {oracle_exact['recall']:.4f}",
        f"Registry oracle IoU>=0.5 recall: {oracle_overlap['recall']:.4f}",
        f"Consensus coverage: {coverage['rate']:.4f} "
        f"({coverage['labels_with_consensus']}/{coverage['total_document_labels']})",
    ])
    if report["alignment_warnings"]:
        lines.append("")
        lines.append("ALIGNMENT WARNINGS")
        lines.extend(f"- {warning}" for warning in report["alignment_warnings"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Labelcentered predictions against Label Studio gold JSON.")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", nargs="*", default=[])
    parser.add_argument("--pred_dir")
    parser.add_argument("--out_json", default="./labelcentered_evaluation.json")
    parser.add_argument("--out_txt", default="./labelcentered_evaluation.txt")
    parser.add_argument("--iou_thresholds", nargs="*", type=float, default=[0.5, 0.75])
    args = parser.parse_args()

    paths = prediction_paths(args.pred, args.pred_dir)
    if not paths:
        raise FileNotFoundError("No prediction JSON files found")
    report = evaluate(
        Path(args.gold).expanduser().resolve(),
        paths,
        args.iou_thresholds,
    )
    out_json = Path(args.out_json).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text = readable_report(report)
    out_txt.write_text(text, encoding="utf-8")
    print(text)
    print(f"JSON report -> {out_json}")
    print(f"Text report -> {out_txt}")
    if report["aligned_record_count"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
