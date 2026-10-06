"""Human-readable inspection for Labelcentered output JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Labelcentered.settings import LABELS


def inspect(payload: dict) -> None:
    print("LABELCENTERED OUTPUT")
    print(f"source_file: {payload.get('source_file')}")
    print(f"overall_consensus_reached: {payload.get('overall_consensus_reached')}")
    print()
    for label in LABELS:
        print(f"{label}")
        print("-" * len(label))
        final_ids = payload.get("final_candidate_ids_by_label", {}).get(label, [])
        recommended_ids = payload.get("recommended_candidate_ids_by_label", {}).get(label, [])
        print(f"  strict final IDs: {final_ids}")
        if recommended_ids != final_ids:
            print(f"  recommended unresolved IDs: {recommended_ids}")
        for span in payload.get("final_spans_by_label", {}).get(label, []):
            print(
                f"  {span['candidate_id']} {span['start']}:{span['end']} "
                f"{span['text']!r} confidence={span.get('confidence_by_expert')}"
            )
            votes = payload.get("candidate_vote_matrices_by_label", {}).get(label, {}).get(span["candidate_id"])
            if votes:
                print(f"    votes: {votes}")
        print(f"  consensus: {payload.get('label_consensus_status', {}).get(label)}")
        print(f"  termination: {payload.get('label_termination_reasons', {}).get(label)}")
        metrics = payload.get("consistency_metrics_by_label", {}).get(label)
        if metrics:
            print(f"  reviewed_by_all: {metrics.get('n_candidates_reviewed_by_all_three')}")
            print(f"  fleiss_kappa: {metrics.get('fleiss_kappa')}")
        print()
    if payload.get("cross_label_conflicts"):
        print("CROSS-LABEL CONFLICTS")
        for conflict in payload["cross_label_conflicts"]:
            print(f"  {conflict}")
    if payload.get("generated_summaries"):
        print("GENERATED SUMMARIES")
        print(
            "  summary_based_on_consensus: "
            f"{payload['generated_summaries'].get('summary_based_on_consensus')}"
        )
        print(
            "  unresolved_labels: "
            f"{payload['generated_summaries'].get('unresolved_labels')}"
        )
        print(payload["generated_summaries"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Labelcentered output JSON.")
    parser.add_argument("json_path")
    args = parser.parse_args()
    inspect(json.loads(Path(args.json_path).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
