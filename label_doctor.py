"""Label-specific DoctorAgent roles and completeness repair."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from Labelcentered.candidate_registry import Candidate
from Labelcentered.candidate_expansion import LABEL_CUES, SourceUnit
from Labelcentered.consistency import normalize_review
from Labelcentered.llm_client import LLMClient, rough_token_estimate
from Labelcentered.rag_retriever import GuidelineRetriever
from Labelcentered.schemas import doctor_review_schema
from Labelcentered.settings import PipelineSettings, role_name


def bounded_excerpt(text: str, max_chars: int, marker: str = " ...[DISPLAY TRUNCATED]... ") -> str:
    if len(text) <= max_chars:
        return text
    side = max(1, (max_chars - len(marker)) // 2)
    return text[:side] + marker + text[-side:]


def candidate_context(source_text: str, candidate: Candidate, window: int = 180) -> str:
    left = source_text[max(0, candidate.start - window):candidate.start]
    right = source_text[candidate.end:min(len(source_text), candidate.end + window)]
    candidate_excerpt = bounded_excerpt(candidate.text, 700)
    return f"{left}[CANDIDATE]{candidate_excerpt}[/CANDIDATE]{right}"


def render_candidate(candidate: Candidate, source_text: str) -> str:
    return (
        f"{candidate.candidate_id} [{candidate.label}] offsets={candidate.start}:{candidate.end} "
        f"experts={','.join(candidate.proposed_by_experts)} "
        f"confidence={candidate.confidence_by_expert} thresholds={candidate.threshold_by_expert}\n"
        f"text_excerpt={bounded_excerpt(candidate.text, 900)!r}\n"
        f"boundary_context={candidate_context(source_text, candidate)!r}"
    )


class LabelDoctor:
    def __init__(
        self,
        *,
        expert_name: str,
        label: str,
        settings: PipelineSettings,
        llm_client: LLMClient,
        retriever: GuidelineRetriever,
    ):
        self.expert_name = expert_name
        self.label = label
        self.settings = settings
        self.llm_client = llm_client
        self.retriever = retriever
        self.agent_role = role_name(expert_name, label)

    def review_candidates(
        self,
        source_text: str,
        candidates: Sequence[Candidate],
        source_units: Sequence[SourceUnit] = (),
    ) -> Dict[str, Any]:
        assigned_ids = [candidate.candidate_id for candidate in candidates]
        responses = []
        remaining = list(candidates)
        attempts = 0
        while remaining and attempts <= self.settings.retry_limit:
            attempts += 1
            batch_responses = []
            for batch in adaptive_candidate_batches(
                remaining,
                source_text,
                self.settings,
            ):
                raw = self._review_batch(source_text, batch, source_units)
                visible_ids = [candidate.candidate_id for candidate in batch]
                normalized = normalize_review(self.agent_role, raw, visible_ids)
                normalized["_visible_candidate_ids"] = visible_ids
                batch_responses.append(normalized)
            responses.extend(batch_responses)
            decided = {
                cid
                for response in responses
                for cid in response.get("decisions_by_candidate", {})
            }
            missing = [candidate for candidate in candidates if candidate.candidate_id not in decided]
            if not missing:
                break
            remaining = missing

        merged = merge_review_batches(self.agent_role, assigned_ids, responses)
        merged["underlying_llm_model"] = self.settings.llm_model
        merged["associated_expert"] = self.expert_name
        return merged

    def _review_batch(
        self,
        source_text: str,
        candidates: Sequence[Candidate],
        source_units: Sequence[SourceUnit] = (),
    ) -> Dict[str, Any]:
        guidelines = "\n".join(self.retriever.retrieve(self.label))
        relevant_units = source_units_for_doctor_batch(
            self.label,
            candidates,
            source_units,
            max_units=doctor_source_unit_limit(self.settings),
        )
        source_unit_text = render_doctor_source_units(
            relevant_units,
            max_chars=doctor_source_unit_char_budget(self.settings),
        )
        system = (
            f"You are {self.agent_role}, an independent PICO annotation reviewer. "
            f"Your associated expert is {self.expert_name}, but all three extractor "
            "models are only evidence sources; do not favor your associated expert. "
            f"Use the RAG guidelines and source context to judge every visible {self.label} "
            "candidate equally. Return keep/reject for each candidate ID, and request "
            "exact quote expansion when the right span is missing or a shown boundary is poor. "
            "Strong proposals matter: copied exact quotes can become new validated candidate IDs."
        )
        user = (
            f"LABEL: {self.label}\nGUIDELINES:\n{guidelines}\n\n"
            "VOTING RULES:\n"
            "- Keep a candidate when its source text fits the label and its boundary is usable, even if it came from a different expert.\n"
            "- Reject wrong-label spans, unsupported text, malformed PDF fragments, and boundaries that are clearly too short or too long.\n"
            "- When rejecting for boundary_too_short, boundary_too_long, or incomplete_phrase, add an exact_quote_proposal for the best complete phrase if visible.\n"
            "- For tiny fragments such as one-word Outcome/ComparisonGroup spans, request an exact_quote_proposal for the complete source phrase when visible in context or source units.\n"
            "- For SampleSize, reject group names without a number; request the exact number-centered span if visible.\n"
            "- If no candidate captures a visible label mention, set missing_label=true and add an expansion request copied exactly from source text.\n"
            "- Prefer requested_operation=exact_quote_proposal with proposed_text copied exactly from a displayed source unit and sentence_id set to that S#### ID.\n"
            "- Do not paraphrase proposed_text; the registry can only validate text copied exactly from the source.\n\n"
            "CANDIDATES:\n" + "\n\n".join(render_candidate(candidate, source_text) for candidate in candidates) +
            "\n\nRELEVANT SOURCE UNITS:\n" + source_unit_text +
            "\n\nReturn JSON with decisions for every visible candidate ID plus any justified expansion_requests."
        )
        visible_ids = [candidate.candidate_id for candidate in candidates]
        return self.llm_client.structured_chat(
            system=system,
            user=user,
            schema=doctor_review_schema(visible_ids, self.label),
            call_name=self.agent_role,
        )



def source_units_for_doctor_batch(
    label: str,
    candidates: Sequence[Candidate],
    source_units: Sequence[SourceUnit],
    *,
    max_units: int = 4,
) -> list[SourceUnit]:
    selected: list[SourceUnit] = []
    seen = set()
    for candidate in candidates:
        for unit in source_units:
            overlaps = max(0, min(candidate.end, unit.end) - max(candidate.start, unit.start))
            if overlaps > 0 and unit.sentence_id not in seen:
                selected.append(unit)
                seen.add(unit.sentence_id)
                break
        if len(selected) >= max_units:
            return selected

    cues = [cue.lower() for cue in LABEL_CUES.get(label, [])]
    for unit in source_units:
        if unit.sentence_id in seen:
            continue
        if any(cue in unit.text.lower() for cue in cues):
            selected.append(unit)
            seen.add(unit.sentence_id)
            if len(selected) >= max_units:
                break
    return selected


def render_doctor_source_units(source_units: Sequence[SourceUnit], *, max_chars: int = 280) -> str:
    if not source_units:
        return "(no compact source units available)"
    return "\n".join(
        f"{unit.sentence_id} {unit.start}:{unit.end} {bounded_excerpt(unit.text, max_chars)!r}"
        for unit in source_units
    )


def doctor_source_unit_limit(settings: PipelineSettings) -> int:
    if settings.llm_context_window <= 8192:
        return 3
    if settings.llm_context_window <= 16384:
        return 4
    return 6


def doctor_source_unit_char_budget(settings: PipelineSettings) -> int:
    if settings.llm_context_window <= 8192:
        return 220
    if settings.llm_context_window <= 16384:
        return 280
    return 420


def merge_review_batches(agent_role: str, assigned_ids: Sequence[str], responses: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    decisions: Dict[str, str] = {}
    expansion_requests: List[Dict[str, Any]] = []
    missing_label = False
    attempts = []
    for response in responses:
        attempts.append(dict(response))
        if not response.get("validation", {}).get("unknown_ids") and not response.get("validation", {}).get("overlap_ids") and not response.get("validation", {}).get("empty_review"):
            decisions.update(response.get("decisions_by_candidate", {}))
        expansion_requests.extend(response.get("expansion_requests", []))
        missing_label = missing_label or bool(response.get("missing_label"))
    keep_ids = [cid for cid in assigned_ids if decisions.get(cid) == "keep"]
    reject_ids = [cid for cid in assigned_ids if decisions.get(cid) == "reject"]
    merged = normalize_review(agent_role, {"keep_ids": keep_ids, "reject_ids": reject_ids, "expansion_requests": expansion_requests, "missing_label": missing_label}, assigned_ids)
    merged["attempts"] = attempts
    return merged


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    batch_size = max(1, batch_size)
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def adaptive_candidate_batches(
    candidates: Sequence[Candidate],
    source_text: str,
    settings: PipelineSettings,
) -> Iterable[Sequence[Candidate]]:
    max_batch_size = max(1, settings.doctor_batch_size)
    output_reserve = min(settings.llm_max_tokens, 2048)
    prompt_budget = max(
        1200,
        settings.llm_context_window
        - output_reserve
        - settings.llm_context_reserve_tokens
        - 700,
    )
    current: List[Candidate] = []
    current_tokens = 0
    for candidate in candidates:
        candidate_tokens = rough_token_estimate(render_candidate(candidate, source_text)) + 24
        if current and (
            len(current) >= max_batch_size
            or current_tokens + candidate_tokens > prompt_budget
        ):
            yield current
            current = []
            current_tokens = 0
        current.append(candidate)
        current_tokens += candidate_tokens
    if current:
        yield current
