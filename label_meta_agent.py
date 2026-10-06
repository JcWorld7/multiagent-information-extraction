"""Seven label-specific MetaAgents."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from Labelcentered.candidate_registry import Candidate
from Labelcentered.consistency import consistency_metrics, evaluate_label_consensus
from Labelcentered.candidate_expansion import LABEL_CUES, SourceUnit
from Labelcentered.label_doctor import bounded_excerpt, candidate_context
from Labelcentered.llm_client import LLMClient
from Labelcentered.rag_retriever import GuidelineRetriever
from Labelcentered.schemas import label_meta_schema
from Labelcentered.settings import PipelineSettings, meta_role_name


class LabelMetaAgent:
    def __init__(self, label: str, settings: PipelineSettings, llm_client: LLMClient, retriever: GuidelineRetriever):
        self.label = label
        self.settings = settings
        self.llm_client = llm_client
        self.retriever = retriever
        self.agent_role = meta_role_name(label)

    def decide(
        self,
        source_text: str,
        candidates: Sequence[Candidate],
        doctor_reviews: Sequence[Dict[str, Any]],
        source_units: Sequence[SourceUnit],
        round_index: int,
        allow_expansion: bool = True,
    ) -> Dict[str, Any]:
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        metrics = consistency_metrics(doctor_reviews, candidate_ids)
        guidelines = "\n".join(self.retriever.retrieve(self.label))
        candidate_contexts = render_meta_candidates(
            source_text,
            candidates,
            total_char_budget=meta_candidate_char_budget(self.settings),
        )
        relevant_source_units = source_units_for_candidates(
            self.label,
            candidates,
            source_units,
            max_units=meta_source_unit_limit(self.settings),
        )
        source_unit_chars = meta_source_unit_char_budget(self.settings)
        relevant_units = "\n".join(
            f"{unit.sentence_id} {unit.start}:{unit.end} "
            f"{bounded_excerpt(unit.text, source_unit_chars)!r}"
            for unit in relevant_source_units
        )
        compact_votes = {
            candidate_id: [
                vote
                for _, vote in sorted(metrics["vote_matrix"].get(candidate_id, {}).items())
            ]
            for candidate_id in candidate_ids
        }
        system = (
            f"You are {self.agent_role}. Treat SciBERT, PubMedBERT, and ClinicalBERT "
            "as equal candidate generators; the doctor votes and source evidence decide. "
            "Select existing validated candidate IDs or request candidate expansion. You "
            "must be source-grounded: prove selected candidates with displayed source "
            "sentences, and actively rescue missing or poorly bounded spans by proposing "
            "exact text from the displayed source units. Do not write final span text or "
            "offsets. The registry, not you, creates any new candidate ID after exact "
            "source verification."
        )
        user = (
            f"LABEL: {self.label}\nGUIDELINES:\n{guidelines}\n\n"
            f"CANDIDATE IDS: {candidate_ids}\n"
            f"CANDIDATE CONTEXTS:\n{candidate_contexts}\n\n"
            f"RELEVANT SOURCE UNITS:\n{relevant_units}\n\n"
            f"DOCTOR VOTES BY CANDIDATE: {compact_votes}\n"
            f"UNANIMOUS KEEP: {metrics['unanimous_keep_candidate_ids']}\n"
            f"UNANIMOUS REJECT: {metrics['unanimous_reject_candidate_ids']}\n"
            f"CONFLICTED: {metrics['conflicted_candidate_ids']}\n"
            f"MISSING DECISIONS: {metrics['missing_decision_candidate_ids']}\n"
            f"ROUND: {round_index}\n"
            "Return candidate-ID arrays using only CANDIDATE IDS. Never place "
            "sentence IDs in candidate-ID arrays. For every selected candidate, add "
            "consensus_proof with evidence_sentence_id from RELEVANT SOURCE UNITS, "
            "an exact evidence_quote copied from that source unit, label-fit rationale, "
            "boundary rationale, and why it is not another label. A useful 2/3 majority "
            "is enough to select when the span is source-grounded and label-correct; "
            "do not require unanimous agreement. Reject one-word fragments when a fuller "
            "nearby phrase is visible. For Outcome, prefer complete measure/instrument "
            "phrases. For SampleSize, prefer number-centered spans. For ComparisonGroup, "
            "prefer complete group phrases. If a majority candidate has boundary problems "
            "or if the label appears missing, inspect RELEVANT SOURCE UNITS and add a "
            "source_search_request. When you find a better boundary or missing span, also "
            "add an expansion request using requested_operation=exact_quote_proposal, "
            "proposed_text copied exactly from the source, and sentence_id exactly one "
            "displayed S#### ID. Use at most five expansion requests. The registry verifies "
            "proposed_text before assigning a new ID."
        )
        if not allow_expansion:
            user += (
                "\nThis is the final review-only pass. Return no expansion "
                "requests; select only currently validated candidate IDs."
            )
        schema = label_meta_schema(
            candidate_ids,
            [unit.sentence_id for unit in relevant_source_units],
            self.label,
            allow_expansion=allow_expansion,
        )
        raw = self.llm_client.structured_chat(
            system=system,
            user=user,
            schema=schema,
            call_name=self.agent_role,
        )
        selected = [cid for cid in raw.get("selected_candidate_ids", []) if cid in set(candidate_ids)]
        selected, removed_redundant = prune_redundant_selected_candidates(candidates, selected)
        rejected = [cid for cid in raw.get("rejected_candidate_ids", []) if cid in set(candidate_ids)]
        expansion_requests = raw.get("expansion_requests", [])
        consensus = evaluate_label_consensus(doctor_reviews, selected, expansion_requests)
        consensus_proof = valid_consensus_proof(
            raw.get("consensus_proof", []),
            selected,
            relevant_source_units,
        )
        source_search_requests = valid_source_search_requests(
            raw.get("source_search_requests", []),
            self.label,
            allow_expansion=allow_expansion,
        )
        return {
            "label": self.label,
            "agent_role": self.agent_role,
            "round": round_index,
            "selected_candidate_ids": selected,
            "redundant_selected_candidate_ids_removed": removed_redundant,
            "rejected_candidate_ids": rejected,
            "expansion_requests": expansion_requests,
            "source_search_requests": source_search_requests,
            "consensus_proof": consensus_proof,
            "unresolved_candidate_ids": sorted(set(raw.get("unresolved_candidate_ids", [])) | set(consensus["unresolved_candidate_ids"])),
            "rationale": raw.get("rationale", ""),
            "consensus": consensus,
            "consistency_metrics": metrics,
        }


def source_units_for_candidates(
    label: str,
    candidates: Sequence[Candidate],
    source_units: Sequence[SourceUnit],
    max_units: int = 12,
) -> list[SourceUnit]:
    selected = []
    seen = set()
    for candidate in candidates:
        for unit in source_units:
            overlaps = max(0, min(candidate.end, unit.end) - max(candidate.start, unit.start))
            if overlaps > 0 and unit.sentence_id not in seen:
                selected.append(unit)
                seen.add(unit.sentence_id)
                break
    cues = [cue.lower() for cue in LABEL_CUES.get(label, [])]
    cue_units = [
        unit
        for unit in source_units
        if unit.sentence_id not in seen
        and any(cue in unit.text.lower() for cue in cues)
    ]
    for unit in cue_units:
        selected.append(unit)
        seen.add(unit.sentence_id)
        if len(selected) >= max_units:
            break
    return selected[:max_units] or list(source_units[:max_units])


def meta_candidate_char_budget(settings: PipelineSettings) -> int:
    if settings.llm_context_window <= 8192:
        return 8000
    if settings.llm_context_window <= 16384:
        return 12000
    return 28000


def meta_source_unit_limit(settings: PipelineSettings) -> int:
    if settings.llm_context_window <= 8192:
        return 6
    if settings.llm_context_window <= 16384:
        return 8
    return 12


def meta_source_unit_char_budget(settings: PipelineSettings) -> int:
    if settings.llm_context_window <= 8192:
        return 300
    if settings.llm_context_window <= 16384:
        return 360
    return 700

def render_meta_candidates(
    source_text: str,
    candidates: Sequence[Candidate],
    total_char_budget: int = 28000,
) -> str:
    if not candidates:
        return "(no current candidates)"
    per_candidate = max(120, min(1100, total_char_budget // len(candidates)))
    text_budget = max(70, int(per_candidate * 0.58))
    context_budget = max(50, per_candidate - text_budget)
    return "\n\n".join(
        f"{candidate.candidate_id}: {bounded_excerpt(candidate.text, text_budget)!r} "
        f"offsets={candidate.start}:{candidate.end}\n"
        f"boundary_context={bounded_excerpt(candidate_context(source_text, candidate), context_budget)!r}"
        for candidate in candidates
    )


def prune_redundant_selected_candidates(
    candidates: Sequence[Candidate],
    selected_ids: Sequence[str],
) -> tuple[list[str], list[str]]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected = [by_id[candidate_id] for candidate_id in selected_ids if candidate_id in by_id]
    groups: list[list[Candidate]] = []
    for candidate in selected:
        for group in groups:
            if any(near_duplicate(candidate, existing) for existing in group):
                group.append(candidate)
                break
        else:
            groups.append([candidate])

    kept = []
    removed = []
    for group in groups:
        best = max(group, key=candidate_boundary_quality)
        kept.append(best.candidate_id)
        removed.extend(candidate.candidate_id for candidate in group if candidate.candidate_id != best.candidate_id)
    kept.sort(key=lambda candidate_id: selected_ids.index(candidate_id))
    return kept, sorted(removed)


def near_duplicate(left: Candidate, right: Candidate) -> bool:
    overlap = max(0, min(left.end, right.end) - max(left.start, right.start))
    shorter = min(left.end - left.start, right.end - right.start)
    if shorter <= 0 or overlap / shorter < 0.9:
        return False
    left_text = normalized_boundary_text(left.text)
    right_text = normalized_boundary_text(right.text)
    return left_text in right_text or right_text in left_text


def normalized_boundary_text(text: str) -> str:
    normalized = " ".join(text.lower().split()).strip(" .,:;-")
    for heading in ("intervention ", "design ", "statistical analysis "):
        if normalized.startswith(heading):
            normalized = normalized[len(heading):]
    return normalized


def candidate_boundary_quality(candidate: Candidate) -> tuple[int, int, int]:
    text = candidate.text.strip()
    score = 0
    if text.endswith((".", "!", "?")):
        score += 5
    if text.startswith((".", ",", ";", ":")):
        score -= 5
    lowered = " ".join(text.lower().split())
    if lowered.startswith(("intervention ", "design ", "statistical analysis ")):
        score -= 2
    if text.endswith(("-", "–", "—")):
        score -= 5
    return score, -len(text), -candidate.start



def valid_consensus_proof(
    rows: Sequence[Dict[str, Any]],
    selected_ids: Sequence[str],
    source_units: Sequence[SourceUnit],
) -> list[Dict[str, Any]]:
    selected = set(selected_ids)
    units = {unit.sentence_id: unit for unit in source_units}
    output = []
    seen = set()
    for row in rows or []:
        candidate_id = str(row.get("candidate_id", ""))
        sentence_id = str(row.get("evidence_sentence_id", ""))
        quote = str(row.get("evidence_quote", ""))
        unit = units.get(sentence_id)
        if candidate_id not in selected or not unit or not quote.strip():
            continue
        if quote not in unit.text:
            continue
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        output.append(
            {
                "candidate_id": candidate_id,
                "evidence_sentence_id": sentence_id,
                "evidence_quote": quote,
                "label_fit_rationale": str(row.get("label_fit_rationale", ""))[:360],
                "boundary_rationale": str(row.get("boundary_rationale", ""))[:360],
                "why_not_other_labels": {
                    str(key): str(value)[:240]
                    for key, value in dict(row.get("why_not_other_labels", {})).items()
                },
                "counterevidence_checked": bool(row.get("counterevidence_checked", False)),
            }
        )
    return output


def valid_source_search_requests(
    rows: Sequence[Dict[str, Any]],
    label: str,
    *,
    allow_expansion: bool,
) -> list[Dict[str, Any]]:
    if not allow_expansion:
        return []
    allowed_operations = {"search_missing_label", "verify_consensus", "repair_boundary"}
    output = []
    for row in rows or []:
        if row.get("label") != label:
            continue
        operation = row.get("requested_operation")
        if operation not in allowed_operations:
            continue
        output.append(
            {
                "label": label,
                "reason": str(row.get("reason", ""))[:300],
                "search_cues": [str(cue)[:80] for cue in row.get("search_cues", [])[:8]],
                "requested_operation": operation,
            }
        )
    return output[:3]
