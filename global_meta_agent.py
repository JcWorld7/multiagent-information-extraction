"""GlobalMetaAgent for assembling frozen label outputs."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from Labelcentered.candidate_registry import CandidateRegistry
from Labelcentered.llm_client import LLMClient
from Labelcentered.schemas import GLOBAL_META_SCHEMA
from Labelcentered.settings import LABELS, PipelineSettings


def detect_cross_label_conflicts(final_spans_by_label: Mapping[str, Sequence[Dict[str, Any]]]) -> list[Dict[str, Any]]:
    conflicts = []
    rows = [
        (label, span)
        for label, spans in final_spans_by_label.items()
        for span in spans
    ]
    for i, (left_label, left) in enumerate(rows):
        for right_label, right in rows[i + 1 :]:
            if left_label == right_label:
                continue
            if left["start"] == right["start"] and left["end"] == right["end"] and left["text"] == right["text"]:
                conflicts.append({"type": "exact_duplicate_cross_label", "labels": [left_label, right_label], "candidate_ids": [left["candidate_id"], right["candidate_id"]]})
            else:
                overlap = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
                shorter = max(1, min(left["end"] - left["start"], right["end"] - right["start"]))
                if overlap / shorter >= 0.8:
                    conflicts.append({"type": "heavy_overlap_cross_label", "labels": [left_label, right_label], "candidate_ids": [left["candidate_id"], right["candidate_id"]], "overlap_chars": overlap})
    return conflicts


class GlobalMetaAgent:
    def __init__(self, settings: PipelineSettings, llm_client: LLMClient | None = None):
        self.settings = settings
        self.llm_client = llm_client

    def assemble(self, registry: CandidateRegistry, final_ids_by_label: Mapping[str, Sequence[str]], label_meta_decisions: Mapping[str, Any]) -> Dict[str, Any]:
        final_spans_by_label = registry.final_spans_by_label(final_ids_by_label)
        conflicts = detect_cross_label_conflicts(final_spans_by_label)
        final_ids = {label: list(final_ids_by_label.get(label, [])) for label in LABELS}
        if self.llm_client is not None and conflicts:
            raw = self.llm_client.structured_chat(
                system="You are GlobalMetaAgent. Select only existing candidate IDs; do not rewrite spans.",
                user=f"FINAL IDS: {final_ids}\nCONFLICTS: {conflicts}\nLABEL META: {label_meta_decisions}",
                schema=GLOBAL_META_SCHEMA,
            )
            final_ids.update({label: [cid for cid in raw.get("final_candidate_ids_by_label", {}).get(label, final_ids[label]) if cid in registry.by_id] for label in LABELS})
            final_spans_by_label = registry.final_spans_by_label(final_ids)
            conflicts = raw.get("cross_label_conflicts", conflicts)
        return {
            "final_candidate_ids_by_label": final_ids,
            "final_spans_by_label": final_spans_by_label,
            "final_spans": [span for label in LABELS for span in final_spans_by_label[label]],
            "cross_label_conflicts": conflicts,
            "rationale": "assembled existing validated candidate IDs without rewriting spans",
        }

