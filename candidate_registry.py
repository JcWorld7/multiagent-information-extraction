"""Append-only label-specific candidate registry."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from Labelcentered.candidate_cleaning import normalize_candidate_span, validate_candidate_text
from Labelcentered.expert_models import RawCandidate
from Labelcentered.settings import LABELS, LABEL_PREFIX


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    label: str
    text: str
    start: int
    end: int
    confidence_by_expert: Dict[str, float]
    threshold_by_expert: Dict[str, float]
    proposed_by_experts: List[str]
    source_section: str
    generation_origin: str
    parent_candidate_id: Optional[str]
    generation_round: int
    validation_status: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_final_span(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence_by_expert": dict(self.confidence_by_expert),
            "threshold_by_expert": dict(self.threshold_by_expert),
            "proposed_by_experts": list(self.proposed_by_experts),
            "source_section": self.source_section,
            "generation_origin": self.generation_origin,
        }


def candidate_key(label: str, text: str, start: int, end: int) -> Tuple[str, str, int, int]:
    return label, text, start, end


class CandidateRegistry:
    def __init__(
        self,
        source_text: str,
        normalize_boundaries: bool = False,
        normalize_candidate_boundaries: bool | None = None,
    ):
        self.source_text = source_text
        if normalize_candidate_boundaries is not None:
            normalize_boundaries = normalize_candidate_boundaries
        self.normalize_boundaries = normalize_boundaries
        self._by_label: Dict[str, List[Candidate]] = {label: [] for label in LABELS}
        self._by_id: Dict[str, Candidate] = {}
        self._keys: Dict[Tuple[str, str, int, int], str] = {}
        self.cleaning_log: List[Dict[str, Any]] = []
        self.registry_expansion_history: List[Dict[str, Any]] = []

    @property
    def by_id(self) -> Dict[str, Candidate]:
        return dict(self._by_id)

    def candidates_for_label(self, label: str) -> List[Candidate]:
        return list(self._by_label[label])

    def registries_json(self) -> Dict[str, List[Dict[str, Any]]]:
        return {label: [candidate.to_dict() for candidate in self._by_label[label]] for label in LABELS}

    def _next_id(self, label: str) -> str:
        return f"{LABEL_PREFIX[label]}-C{len(self._by_label[label]) + 1:04d}"

    def add_candidate(
        self,
        *,
        label: str,
        text: str,
        start: int,
        end: int,
        confidence_by_expert: Optional[Mapping[str, float]] = None,
        threshold_by_expert: Optional[Mapping[str, float]] = None,
        proposed_by_experts: Optional[Iterable[str]] = None,
        source_section: str = "unknown",
        generation_origin: str = "expert",
        parent_candidate_id: Optional[str] = None,
        generation_round: int = 0,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Candidate], str]:
        provenance_dict = dict(provenance or {})
        if self.normalize_boundaries:
            normalized_text, normalized_start, normalized_end, normalization = normalize_candidate_span(
                source_text=self.source_text,
                label=label,
                text=text,
                start=start,
                end=end,
            )
            if normalization.get("changed"):
                provenance_dict["boundary_normalization"] = normalization
                text, start, end = normalized_text, normalized_start, normalized_end
        decision = validate_candidate_text(
            source_text=self.source_text,
            label=label,
            text=text,
            start=start,
            end=end,
        )
        log_entry = {
            "label": label,
            "text": text,
            "start": start,
            "end": end,
            "generation_origin": generation_origin,
            "decision": decision.to_dict(),
        }
        if not decision.accepted:
            self.cleaning_log.append(log_entry)
            return None, decision.reason

        key = candidate_key(label, text, start, end)
        duplicate_id = self._keys.get(key)
        if duplicate_id:
            existing = self._by_id[duplicate_id]
            if generation_origin == "expert":
                merged = self._merge_expert_duplicate(
                    existing,
                    confidence_by_expert or {},
                    threshold_by_expert or {},
                    proposed_by_experts or [],
                    provenance_dict,
                )
                self._replace_candidate(existing, merged)
                return merged, "merged_duplicate"
            self.cleaning_log.append({**log_entry, "decision": {"accepted": False, "reason": "duplicate_exact_span"}})
            return None, "duplicate_exact_span"

        candidate = Candidate(
            candidate_id=self._next_id(label),
            label=label,
            text=text,
            start=start,
            end=end,
            confidence_by_expert=dict(confidence_by_expert or {}),
            threshold_by_expert=dict(threshold_by_expert or {}),
            proposed_by_experts=sorted(set(proposed_by_experts or [])),
            source_section=source_section,
            generation_origin=generation_origin,
            parent_candidate_id=parent_candidate_id,
            generation_round=generation_round,
            validation_status="validated",
            provenance=provenance_dict,
        )
        self._by_label[label].append(candidate)
        self._by_id[candidate.candidate_id] = candidate
        self._keys[key] = candidate.candidate_id
        self.cleaning_log.append({**log_entry, "candidate_id": candidate.candidate_id})
        if generation_origin != "expert":
            self.registry_expansion_history.append(candidate.to_dict())
        return candidate, "accepted"

    def _merge_expert_duplicate(
        self,
        existing: Candidate,
        confidence_by_expert: Mapping[str, float],
        threshold_by_expert: Mapping[str, float],
        proposed_by_experts: Iterable[str],
        provenance: Mapping[str, Any],
    ) -> Candidate:
        confidences = dict(existing.confidence_by_expert)
        for expert, value in confidence_by_expert.items():
            confidences[expert] = max(confidences.get(expert, value), value)
        thresholds = dict(existing.threshold_by_expert)
        thresholds.update(threshold_by_expert)
        experts = sorted(set(existing.proposed_by_experts) | set(proposed_by_experts))
        merged_provenance = dict(existing.provenance)
        merged_provenance.setdefault("merged_duplicates", []).append(dict(provenance))
        return Candidate(
            candidate_id=existing.candidate_id,
            label=existing.label,
            text=existing.text,
            start=existing.start,
            end=existing.end,
            confidence_by_expert=confidences,
            threshold_by_expert=thresholds,
            proposed_by_experts=experts,
            source_section=existing.source_section,
            generation_origin=existing.generation_origin,
            parent_candidate_id=existing.parent_candidate_id,
            generation_round=existing.generation_round,
            validation_status=existing.validation_status,
            provenance=merged_provenance,
        )

    def _replace_candidate(self, old: Candidate, new: Candidate) -> None:
        rows = self._by_label[old.label]
        for index, candidate in enumerate(rows):
            if candidate.candidate_id == old.candidate_id:
                rows[index] = new
                break
        self._by_id[new.candidate_id] = new

    def build_from_raw(self, raw_by_expert_and_label: Mapping[str, Mapping[str, Sequence[RawCandidate]]]) -> None:
        for expert, by_label in raw_by_expert_and_label.items():
            for label in LABELS:
                for raw in by_label.get(label, []):
                    self.add_candidate(
                        label=raw.label,
                        text=raw.text,
                        start=raw.start,
                        end=raw.end,
                        confidence_by_expert={expert: raw.expert_confidence},
                        threshold_by_expert={expert: raw.threshold_used},
                        proposed_by_experts=[expert],
                        source_section=raw.source_section,
                        generation_origin="expert",
                        provenance=raw.provenance,
                    )

    def final_spans_for_ids(self, candidate_ids: Iterable[str]) -> List[Dict[str, Any]]:
        out = []
        for candidate_id in candidate_ids:
            candidate = self._by_id[candidate_id]
            out.append(candidate.to_final_span())
        return out

    def final_spans_by_label(self, final_ids_by_label: Mapping[str, Sequence[str]]) -> Dict[str, List[Dict[str, Any]]]:
        return {
            label: self.final_spans_for_ids(final_ids_by_label.get(label, []))
            for label in LABELS
        }

