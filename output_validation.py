"""Output validation for label-centered extraction JSON."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

from Labelcentered.settings import LABELS


def validate_output(payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    source_text = payload.get("model_input_text", "")
    registry = {
        candidate["candidate_id"]: candidate
        for rows in payload.get("label_candidate_registries", {}).values()
        for candidate in rows
    }
    if set(payload.get("final_spans_by_label", {}).keys()) != set(LABELS):
        errors.append("final_spans_by_label must contain all seven labels")
    strict_policy = payload.get("final_selection_policy", "strict") == "strict"
    for label in LABELS:
        if label not in payload.get("final_candidate_ids_by_label", {}):
            errors.append(f"final_candidate_ids_by_label missing {label}")
        vote_matrix = payload.get("candidate_vote_matrices_by_label", {}).get(label, {})
        for candidate_id in payload.get("final_candidate_ids_by_label", {}).get(label, []):
            votes = list(vote_matrix.get(candidate_id, {}).values())
            if strict_policy and vote_matrix and (len(votes) != 3 or any(vote != "keep" for vote in votes)):
                errors.append(
                    f"final candidate {candidate_id} is not supported by three keep votes"
                )
    flat = []
    for label, spans in payload.get("final_spans_by_label", {}).items():
        if label not in LABELS:
            errors.append(f"unsupported label in final_spans_by_label: {label}")
        for span in spans:
            flat.append(span)
            candidate_id = span.get("candidate_id")
            if candidate_id not in registry:
                errors.append(f"final span candidate_id not in registry: {candidate_id}")
            if source_text:
                start, end, text = span.get("start"), span.get("end"), span.get("text")
                if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or source_text[start:end] != text:
                    errors.append(f"invalid final span offsets/text for {candidate_id}")
    flat_ids = [span.get("candidate_id") for span in payload.get("final_spans", [])]
    by_label_ids = [span.get("candidate_id") for span in flat]
    if flat_ids != by_label_ids:
        errors.append("final_spans_by_label does not agree with final_spans ordering/content")
    if payload.get("overall_consensus_reached") and payload.get("unresolved_candidates"):
        errors.append("overall_consensus_reached cannot be true with unresolved_candidates")
    consensus_by_label = payload.get("label_consensus_status", {})
    final_ids_by_label = payload.get("final_candidate_ids_by_label", {})
    for label in LABELS:
        if consensus_by_label.get(label) is True and not final_ids_by_label.get(label):
            errors.append(f"{label} cannot have consensus=true with no selected candidates")
    summaries = payload.get("generated_summaries")
    if summaries and any("start" in value for value in summaries.values() if isinstance(value, dict)):
        errors.append("generated_summaries must not replace exact spans")
    if summaries and not (
        summaries.get("summary_based_on_final_spans") is True
        or summaries.get("summary_based_on_strict_unanimous_spans") is True
    ):
        errors.append("generated_summaries must declare final-span provenance")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Labelcentered output JSON.")
    parser.add_argument("json_path")
    args = parser.parse_args()
    path = Path(args.json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_output(payload)
    if errors:
        for error in errors:
            print(f"[error] {error}")
        sys.exit(1)
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
