"""Deterministic append-only candidate expansion."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from Labelcentered.candidate_registry import Candidate, CandidateRegistry
from Labelcentered.settings import LABELS


@dataclass(frozen=True)
class SourceUnit:
    sentence_id: str
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def segment_source_units(source_text: str) -> List[SourceUnit]:
    units = []
    pattern = re.compile(r".+?(?:[.!?](?=\s+|$)|\n{2,}|$)", re.DOTALL)
    for match in pattern.finditer(source_text):
        raw = match.group(0)
        if not raw.strip():
            continue
        start = match.start() + (len(raw) - len(raw.lstrip()))
        end = match.start() + len(raw.rstrip())
        if end > start:
            units.append(SourceUnit(f"S{len(units) + 1:04d}", start, end, source_text[start:end]))
    if not units and source_text:
        units.append(SourceUnit("S0001", 0, len(source_text), source_text))
    return units


LABEL_CUES = {
    "Participant": ["participants", "students", "teachers", "children", "adolescents", "eligibility", "eligible", "aged", "recruited"],
    "Intervention": ["programme", "program", "intervention", "treatment", "therapy", "sessions", "training", "curriculum", "counseling", "mindfulness", "CBT", "SEL"],
    "Outcome": ["outcome", "endpoint", "measure", "measured", "assessed", "scale", "score", "symptoms", "self-esteem", "stress", "depression", "anxiety", "well-being"],
    "SampleSize": ["n=", "n =", "N=", "N =", "participants", "subjects", "students", "teachers", "enrolled", "completed", "allocated", "randomized", "sample"],
    "ComparisonGroup": ["control", "wait-list", "waitlist", "usual care", "placebo", "comparison", "business-as-usual", "BAU"],
    "DesignDescription": [
        "randomized",
        "randomised",
        "randomly assigned",
        "assigned to",
        "allocation",
        "quasi-experimental",
        "cohort",
        "trial",
        "cross-sectional",
    ],
    "StatisticalAnalysis": ["regression", "ANCOVA", "ANOVA", "mixed-effects", "t-test", "subgroup analysis", "Stata"],
}


PRE_EXPANSION_LABELS = {"Participant", "Intervention", "Outcome", "SampleSize"}


def deterministic_pre_expand_registry(
    registry: CandidateRegistry,
    source_units: Sequence[SourceUnit],
    *,
    labels: Sequence[str] | None = None,
    max_new_per_label: int = 80,
) -> List[Candidate]:
    """Append exact source-grounded candidates before LLM review.

    This raises the candidate-registry recall ceiling without asking the LLM to
    invent text or offsets. Every candidate is still validated by the registry.
    """
    active_labels = [label for label in (labels or list(PRE_EXPANSION_LABELS)) if label in PRE_EXPANSION_LABELS]
    added: List[Candidate] = []
    for label in active_labels:
        before = len(added)
        for start, end, method in deterministic_label_spans(registry, source_units, label):
            if len(added) - before >= max_new_per_label:
                break
            candidate = append_pre_expansion_candidate(registry, label, start, end, method)
            if candidate:
                added.append(candidate)
    return added


def deterministic_label_spans(
    registry: CandidateRegistry,
    source_units: Sequence[SourceUnit],
    label: str,
) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    spans.extend(pattern_spans(registry.source_text, label))
    spans.extend(clause_boundary_alternatives(registry, source_units, label))
    spans.extend(candidate_internal_alternatives(registry, label))
    return deduplicate_spans(spans)


def pattern_spans(source_text: str, label: str) -> List[Tuple[int, int, str]]:
    patterns = LABEL_PRE_EXPANSION_PATTERNS.get(label, [])
    spans: List[Tuple[int, int, str]] = []
    for name, pattern in patterns:
        for match in pattern.finditer(source_text):
            start, end = trim_generated_span(source_text, match.start(), match.end())
            if end > start:
                spans.append((start, end, name))
    return spans


def clause_boundary_alternatives(
    registry: CandidateRegistry,
    source_units: Sequence[SourceUnit],
    label: str,
) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cue_regex = LABEL_CUE_REGEX[label]
    for unit in source_units:
        if not cue_regex.search(unit.text):
            continue
        for start, end in clause_spans(registry.source_text, unit.start, unit.end):
            text = registry.source_text[start:end]
            if cue_regex.search(text) and generated_span_length_ok(label, text):
                spans.append((start, end, "cue_clause_boundary"))
    return spans


def candidate_internal_alternatives(registry: CandidateRegistry, label: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for candidate in registry.candidates_for_label(label):
        if len(candidate.text) < 45:
            continue
        for _, pattern in LABEL_PRE_EXPANSION_PATTERNS.get(label, []):
            for match in pattern.finditer(candidate.text):
                start = candidate.start + match.start()
                end = candidate.start + match.end()
                start, end = trim_generated_span(registry.source_text, start, end)
                if end > start and (start, end) != (candidate.start, candidate.end):
                    spans.append((start, end, "candidate_internal_boundary_alternative"))
        for start, end in clause_spans(registry.source_text, candidate.start, candidate.end):
            if (start, end) != (candidate.start, candidate.end):
                text = registry.source_text[start:end]
                if LABEL_CUE_REGEX[label].search(text) and generated_span_length_ok(label, text):
                    spans.append((start, end, "candidate_clause_boundary_alternative"))
    return spans


def append_pre_expansion_candidate(
    registry: CandidateRegistry,
    label: str,
    start: int,
    end: int,
    method: str,
) -> Optional[Candidate]:
    text = registry.source_text[start:end]
    candidate, _ = registry.add_candidate(
        label=label,
        text=text,
        start=start,
        end=end,
        confidence_by_expert={},
        threshold_by_expert={},
        proposed_by_experts=[],
        source_section=source_section_for_offset(registry.source_text, start),
        generation_origin="deterministic_pre_expansion",
        generation_round=0,
        provenance={"generation_method": method, "validation_method": "exact_source_offsets"},
    )
    return candidate


def clause_spans(source_text: str, start: int, end: int) -> List[Tuple[int, int]]:
    text = source_text[start:end]
    spans: List[Tuple[int, int]] = []
    offset = 0
    for piece in re.split(r"([.;:])", text):
        piece_start = start + offset
        offset += len(piece)
        if piece in ".;:" or not piece.strip():
            continue
        left, right = trim_generated_span(source_text, piece_start, piece_start + len(piece))
        if right > left:
            spans.append((left, right))
    return spans


def trim_generated_span(source_text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and source_text[start].isspace():
        start += 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    while start < end and source_text[start] in ",;:)]}”’\"'":
        start += 1
    while end > start and source_text[end - 1] in ",;:([{“‘\"'":
        end -= 1
    return start, end


def generated_span_length_ok(label: str, text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
    if label == "SampleSize":
        return 1 <= len(words) <= 18 and bool(re.search(r"\d", text))
    if label == "Outcome":
        return 2 <= len(words) <= 18
    if label == "Intervention":
        return 2 <= len(words) <= 24
    if label == "Participant":
        return 2 <= len(words) <= 22
    return True


def deduplicate_spans(spans: Sequence[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    output: List[Tuple[int, int, str]] = []
    seen = set()
    for start, end, method in sorted(spans, key=lambda row: (row[0], row[1], row[2])):
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        output.append((start, end, method))
    return output


def source_section_for_offset(source_text: str, start: int) -> str:
    methods = source_text.find("=== METHODS ===")
    results = source_text.find("=== RESULTS ===")
    if results != -1 and start >= results:
        return "results"
    if methods != -1 and start >= methods:
        return "methods"
    return "unknown"


def compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.DOTALL)


UNIT = r"participants?|students?|teachers?|children|adolescents?|schools?|classrooms?|parents?|families|subjects?|pupils|youths?|dyads?|clusters?"

LABEL_PRE_EXPANSION_PATTERNS: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {
    "SampleSize": [
        ("sample_n_equals", compile_pattern(rf"\b[Nn]\s*=\s*\d[\d,]*(?:\.\d+)?(?:\s*{UNIT})?\b")),
        ("sample_number_unit", compile_pattern(rf"\b\d[\d,]*(?:\.\d+)?\s+{UNIT}\b")),
        ("sample_group_n", compile_pattern(r"\b(?:intervention|control|comparison|treatment|wait[-\s]?list|usual\s+care|experimental)(?:\s+group)?\s+n\s*=\s*\d[\d,]*(?:\.\d+)?\b")),
    ],
    "Participant": [
        ("participant_group_phrase", compile_pattern(rf"\b(?:[A-Za-z0-9%/+-]+\s+){{0,8}}{UNIT}\b")),
        ("participant_aged_phrase", compile_pattern(rf"\b{UNIT}\s+(?:aged|ages?)\s+\d{{1,2}}(?:\s*(?:-|to)\s*\d{{1,2}})?\b")),
        ("participant_grade_phrase", compile_pattern(r"\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|middle|high|secondary|primary|elementary)\s+(?:grade\s+)?(?:students|children|pupils|adolescents)\b")),
    ],
    "Intervention": [
        ("intervention_named_program", compile_pattern(r"\b(?:the\s+)?(?:[A-Z][A-Za-z0-9'&-]+|[a-z]+)(?:\s+(?:[A-Z][A-Za-z0-9'&-]+|[a-z]+)){0,8}\s+(?:program|programme|curriculum|intervention|therapy|training)\b")),
        ("intervention_session_phrase", compile_pattern(r"\b\d+\s+(?:weekly\s+)?(?:sessions|lessons|meetings|classes|modules)\b(?:\s+(?:of|focused\s+on|covering)\s+[A-Za-z][^.;,]{0,80})?")),
        ("intervention_cue_phrase", compile_pattern(r"\b(?:mindfulness|CBT|cognitive[-\s]behavioral|social[-\s]emotional|SEL|counseling|coping|skills?)\b[^.;]{0,120}\b(?:program|programme|curriculum|sessions?|lessons?|training|intervention|therapy)\b")),
    ],
    "Outcome": [
        ("outcome_symptom_phrase", compile_pattern(r"\b(?:anxiety|depression|depressive|stress|trauma|internalizing|externalizing|conduct|behavioral|behavioural|emotional|prosocial|self[-\s]esteem|well[-\s]?being|resilience|coping)\s+(?:symptoms?|scores?|problems?|behavio[u]?rs?|skills?|well[-\s]?being|scale|subscale|measure|outcome)\b")),
        ("outcome_measure_phrase", compile_pattern(r"\b(?:[A-Za-z][A-Za-z'&-]+\s+){0,8}(?:scale|subscale|inventory|questionnaire|assessment|index|measure|score)\b")),
        ("outcome_instrument_acronym", compile_pattern(r"\b(?:CES-D|SDQ|RCADS|BDI|CBCL|YSR|PSS|RSES)(?:\s+(?:scale|score|subscale|inventory|questionnaire))?\b")),
    ],
}

LABEL_CUE_REGEX: Dict[str, re.Pattern[str]] = {
    label: compile_pattern(r"\b(?:" + "|".join(re.escape(cue) for cue in cues) + r")\b")
    for label, cues in LABEL_CUES.items()
}


class CandidateExpansionManager:
    def __init__(self, registry: CandidateRegistry, source_units: Sequence[SourceUnit]):
        self.registry = registry
        self.source_units = list(source_units)
        self.resolved_proposals: List[Dict[str, Any]] = []
        self.unresolved_proposals: List[Dict[str, Any]] = []
        self.last_request_outcomes: List[Dict[str, Any]] = []
        self._request_history: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def handle_requests(
        self,
        requests: Iterable[Mapping[str, Any]],
        *,
        proposed_by_agent: str,
        generation_round: int,
    ) -> List[Candidate]:
        added = []
        self.last_request_outcomes = []
        for request in requests:
            request_dict = dict(request)
            signature = request_signature(request_dict)
            previous = self._request_history.get(signature)
            if previous and previous.get("resolved"):
                outcome = {
                    "request": request_dict,
                    "resolved": True,
                    "status": "previously_resolved",
                    "candidate_id": previous.get("candidate_id"),
                    "reason": previous.get("reason"),
                }
                self.last_request_outcomes.append(outcome)
                self.resolved_proposals.append(outcome)
                continue
            if previous and not previous.get("resolved"):
                outcome = {
                    "request": request_dict,
                    "resolved": False,
                    "status": "previously_unresolved",
                    "candidate_id": previous.get("candidate_id"),
                    "reason": previous.get("reason"),
                }
                self.last_request_outcomes.append(outcome)
                continue
            before_resolved = len(self.resolved_proposals)
            before_unresolved = len(self.unresolved_proposals)
            new_candidates = self._handle_one(request_dict, proposed_by_agent, generation_round)
            added.extend(new_candidates)
            if len(self.resolved_proposals) > before_resolved:
                record = dict(self.resolved_proposals[-1])
                outcome = {
                    **record,
                    "resolved": True,
                    "status": record.get("status", "candidate_appended"),
                }
            elif len(self.unresolved_proposals) > before_unresolved:
                record = dict(self.unresolved_proposals[-1])
                outcome = {
                    **record,
                    "resolved": False,
                    "status": "unresolved",
                }
            else:
                outcome = {
                    "request": request_dict,
                    "resolved": False,
                    "status": "unresolved",
                    "reason": "no_request_outcome",
                }
                self.unresolved_proposals.append(outcome)
            self.last_request_outcomes.append(outcome)
            self._request_history[signature] = outcome
        return added

    def _handle_one(self, request: Dict[str, Any], proposed_by_agent: str, generation_round: int) -> List[Candidate]:
        label = request.get("label")
        if label not in LABELS:
            self._unresolved(request, "unsupported_label")
            return []
        operation = request.get("requested_operation", "")
        if operation == "exact_quote_proposal" or request.get("proposed_text"):
            candidate = self._exact_quote(request, proposed_by_agent, generation_round)
            return [candidate] if candidate else []
        if operation in {"split_by_sentence", "split_by_clause"}:
            return self._split_parent(request, proposed_by_agent, generation_round)
        if operation in {"expand_to_sentence_start", "expand_to_sentence_end"}:
            candidate = self._expand_parent(request, proposed_by_agent, generation_round)
            return [candidate] if candidate else []
        if operation in {"generate_number_centered_span", "search_missing_label"}:
            return self._search_missing_label(request, proposed_by_agent, generation_round)
        self._unresolved(request, "unsupported_operation")
        return []

    def _exact_quote(self, request: Dict[str, Any], proposed_by_agent: str, generation_round: int) -> Optional[Candidate]:
        quote = str(request.get("proposed_text", ""))
        if not quote.strip():
            self._unresolved(request, "empty_proposed_text")
            return None
        regions = self._search_regions(request)
        hits = []
        for start, end in regions:
            window = self.registry.source_text[start:end]
            for match in re.finditer(re.escape(quote), window):
                hits.append((start + match.start(), start + match.end()))
        hits = sorted(set(hits))
        if not hits:
            self._unresolved(request, "proposed_text_not_found")
            return None
        if len(hits) > 1:
            self._unresolved(request, "ambiguous_proposed_text")
            return None
        start, end = hits[0]
        return self._append(request, start, end, proposed_by_agent, generation_round, "exact_quote_proposal", "bounded_exact_quote_search")

    def _search_regions(self, request: Mapping[str, Any]) -> List[Tuple[int, int]]:
        sentence_id = request.get("sentence_id")
        if sentence_id:
            return [(unit.start, unit.end) for unit in self.source_units if unit.sentence_id == sentence_id]
        parent_id = request.get("parent_candidate_id")
        if parent_id and parent_id in self.registry.by_id:
            parent = self.registry.by_id[parent_id]
            return [(max(0, parent.start - 300), min(len(self.registry.source_text), parent.end + 300))]
        return [(0, len(self.registry.source_text))]

    def _split_parent(self, request: Dict[str, Any], proposed_by_agent: str, generation_round: int) -> List[Candidate]:
        parent = self._parent(request)
        if not parent:
            return []
        resolved_before = len(self.resolved_proposals)
        spans = []
        if request.get("requested_operation") == "split_by_sentence":
            spans = [(unit.start, unit.end) for unit in self.source_units if parent.start <= unit.start and unit.end <= parent.end]
        else:
            text = self.registry.source_text[parent.start:parent.end]
            offset = 0
            for piece in re.split(r"([,;:])", text):
                piece_start = parent.start + offset
                offset += len(piece)
                if piece in ",;:" or not piece.strip():
                    continue
                spans.append((piece_start, piece_start + len(piece)))
        added = []
        for start, end in spans:
            candidate = self._append(request, start, end, proposed_by_agent, generation_round, request.get("requested_operation", ""), "deterministic_parent_split")
            if candidate:
                added.append(candidate)
        if not added and len(self.resolved_proposals) == resolved_before:
            self._unresolved(request, "no_valid_split_candidate")
        return added

    def _expand_parent(self, request: Dict[str, Any], proposed_by_agent: str, generation_round: int) -> Optional[Candidate]:
        parent_id = request.get("parent_candidate_id")
        parent = self.registry.by_id.get(parent_id) if parent_id else None
        sentence_id = request.get("sentence_id")
        source_unit = self._source_unit(sentence_id) if sentence_id else None

        if not parent:
            if source_unit:
                sentence_request = dict(request)
                sentence_request["parent_candidate_id"] = None
                return self._append(
                    sentence_request,
                    source_unit.start,
                    source_unit.end,
                    proposed_by_agent,
                    generation_round,
                    request.get("requested_operation", ""),
                    "validated_source_unit_sentence_recovery",
                    generation_origin="missing_label_recovery",
                    provenance_extra={
                        "sentence_id": source_unit.sentence_id,
                        "ignored_parent_candidate_id": parent_id,
                    },
                )
            if sentence_id:
                self._unresolved(request, "unknown_sentence_id")
            else:
                self._unresolved(request, "unknown_parent_candidate_id")
            return None

        if sentence_id and not source_unit:
            self._unresolved(request, "unknown_sentence_id")
            return None
        if source_unit and not (source_unit.start <= parent.start and parent.end <= source_unit.end):
            self._unresolved(request, "parent_not_inside_requested_source_unit")
            return None

        candidate_units = [source_unit] if source_unit else self.source_units
        for unit in candidate_units:
            if unit.start <= parent.start and parent.end <= unit.end:
                start = unit.start if request.get("requested_operation") == "expand_to_sentence_start" else parent.start
                end = unit.end if request.get("requested_operation") == "expand_to_sentence_end" else parent.end
                return self._append(request, start, end, proposed_by_agent, generation_round, request.get("requested_operation", ""), "containing_sentence_expansion")
        self._unresolved(request, "parent_not_inside_source_unit")
        return None

    def _source_unit(self, sentence_id: Any) -> Optional[SourceUnit]:
        if not sentence_id:
            return None
        reference = str(sentence_id).strip()
        exact = next(
            (unit for unit in self.source_units if unit.sentence_id == reference),
            None,
        )
        if exact:
            return exact

        id_match = re.search(r"(?<![A-Za-z0-9])S\d{4}(?!\d)", reference, flags=re.IGNORECASE)
        if id_match:
            normalized_id = id_match.group(0).upper()
            return next(
                (unit for unit in self.source_units if unit.sentence_id == normalized_id),
                None,
            )

        if reference.isdigit():
            offset = int(reference)
            return next(
                (
                    unit
                    for unit in self.source_units
                    if unit.start == offset or unit.start <= offset < unit.end
                ),
                None,
            )
        return None

    def _search_missing_label(self, request: Dict[str, Any], proposed_by_agent: str, generation_round: int) -> List[Candidate]:
        label = request["label"]
        added = []
        resolved_before = len(self.resolved_proposals)
        if label == "SampleSize":
            pattern = re.compile(r"\b(?:[Nn]\s*=\s*)?\d{1,5}\s+(?:participants|subjects|patients|students|adolescents|adults)\b|\b[Nn]\s*=\s*\d{1,5}\b")
            for match in pattern.finditer(self.registry.source_text):
                candidate = self._append(request, match.start(), match.end(), proposed_by_agent, generation_round, "generate_number_centered_span", "deterministic_sample_size_pattern")
                if candidate:
                    added.append(candidate)
        if added:
            return added
        cues = [cue.lower() for cue in LABEL_CUES.get(label, [])]
        for unit in self.source_units:
            if any(cue in unit.text.lower() for cue in cues):
                candidate = self._append(request, unit.start, unit.end, proposed_by_agent, generation_round, "search_missing_label", "deterministic_label_cue_sentence")
                if candidate:
                    return [candidate]
                if len(self.resolved_proposals) > resolved_before:
                    return []
        self._unresolved(request, "no_missing_label_candidate_found")
        return []

    def _parent(self, request: Mapping[str, Any]) -> Optional[Candidate]:
        parent_id = request.get("parent_candidate_id")
        if not parent_id or parent_id not in self.registry.by_id:
            self._unresolved(dict(request), "unknown_parent_candidate_id")
            return None
        return self.registry.by_id[parent_id]

    def _append(
        self,
        request: Dict[str, Any],
        start: int,
        end: int,
        proposed_by_agent: str,
        generation_round: int,
        method: str,
        validation_method: str,
        *,
        generation_origin: str = "registry_expansion",
        provenance_extra: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Candidate]:
        text = self.registry.source_text[start:end]
        provenance = {
            "proposed_by_agent": proposed_by_agent,
            "generation_method": method,
            "validation_method": validation_method,
            "request": request,
        }
        provenance.update(dict(provenance_extra or {}))
        candidate, reason = self.registry.add_candidate(
            label=request["label"],
            text=text,
            start=start,
            end=end,
            confidence_by_expert={},
            threshold_by_expert={},
            proposed_by_experts=[],
            source_section="expanded",
            generation_origin=generation_origin,
            parent_candidate_id=request.get("parent_candidate_id") or None,
            generation_round=generation_round,
            provenance=provenance,
        )
        record = {"request": request, "reason": reason, "candidate_id": candidate.candidate_id if candidate else None}
        if candidate:
            self.resolved_proposals.append({**record, "status": "candidate_appended"})
        elif reason == "duplicate_exact_span":
            existing = self._existing_candidate(request["label"], text, start, end)
            self.resolved_proposals.append(
                {
                    **record,
                    "status": "already_exists",
                    "candidate_id": existing.candidate_id if existing else None,
                    "resolved": True,
                }
            )
        else:
            self.unresolved_proposals.append(record)
        return candidate

    def _existing_candidate(
        self,
        label: str,
        text: str,
        start: int,
        end: int,
    ) -> Optional[Candidate]:
        return next(
            (
                candidate
                for candidate in self.registry.candidates_for_label(label)
                if candidate.text == text
                and candidate.start == start
                and candidate.end == end
            ),
            None,
        )

    def _unresolved(self, request: Dict[str, Any], reason: str) -> None:
        self.unresolved_proposals.append({"request": request, "reason": reason})


def request_signature(request: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        request.get("label"),
        request.get("requested_operation"),
        request.get("parent_candidate_id"),
        request.get("sentence_id"),
        request.get("proposed_text"),
    )
