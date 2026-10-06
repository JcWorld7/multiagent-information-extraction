"""Optional summary generation after exact extraction is frozen."""
from __future__ import annotations

from typing import Any, Dict, Mapping

from Labelcentered.llm_client import LLMClient
from Labelcentered.schemas import SUMMARY_SCHEMA


class SummaryAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def summarize(
        self,
        final_spans_by_label: Mapping[str, Any],
        *,
        overall_consensus_reached: bool,
        label_consensus_status: Mapping[str, bool],
        final_selection_policy: str = "strict",
    ) -> Dict[str, Any]:
        unresolved_labels = [
            label
            for label, reached in label_consensus_status.items()
            if not reached
        ]
        result = self.llm_client.structured_chat(
            system=(
                "Summarize the frozen exact PICO extraction. You may paraphrase in "
                "the summary, but must not change candidate IDs, labels, offsets, "
                "or final spans. The input contains only already-selected final "
                "registry spans. Do not infer missing labels."
            ),
            user=(
                f"OVERALL CONSENSUS: {overall_consensus_reached}\n"
                f"FINAL SELECTION POLICY: {final_selection_policy}\n"
                f"UNRESOLVED LABELS: {unresolved_labels}\n"
                f"FROZEN FINAL SPANS BY LABEL:\n{final_spans_by_label}"
            ),
            schema=SUMMARY_SCHEMA,
            call_name="SummaryAgent",
        )
        result["summary_based_on_consensus"] = overall_consensus_reached
        result["summary_based_on_final_spans"] = True
        result["summary_final_selection_policy"] = final_selection_policy
        result["summary_based_on_strict_unanimous_spans"] = final_selection_policy == "strict"
        result["unresolved_labels"] = unresolved_labels
        return result
