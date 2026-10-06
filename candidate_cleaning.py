"""Deterministic candidate cleaning and validation rules."""
from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from Labelcentered.settings import LABELS

STOPWORDS = {"the", "a", "an", "of", "in", "and", "or", "to", "for"}
VALID_LABELS = set(LABELS)
INCOMPLETE_TRAILING_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "past",
    "the",
    "to",
    "with",
}
STATISTICAL_METHOD_CUES = {
    "ancova",
    "anova",
    "regression",
    "mixed-effects",
    "mixed effects",
    "t-test",
    "t-tests",
    "chi-square",
    "imputation",
    "f-test",
    "f-tests",
}


@dataclass
class CleaningDecision:
    accepted: bool
    reason: str
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def is_punctuation_only(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(ch in string.punctuation for ch in stripped)


def is_isolated_stopword(text: str) -> bool:
    return text.strip().lower() in STOPWORDS


def is_one_character_fragment(label: str, text: str) -> bool:
    stripped = text.strip()
    if label == "SampleSize" and stripped.isdigit():
        return False
    return len(stripped) == 1 and stripped.isalpha()


def looks_like_broken_subword(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("##"):
        return True
    if re.fullmatch(r"[-\u2010-\u2015]?[A-Za-z]{1,2}[-\u2010-\u2015]?", stripped):
        return True
    return bool(
        len(stripped) <= 20
        and " " not in stripped
        and re.fullmatch(r"[A-Za-z]{2,}[-\u2010-\u2015]", stripped)
    )


def looks_like_malformed_pdf_fragment(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) <= 2 and not stripped.isdigit():
        return True
    if (
        len(stripped) <= 50
        and len(stripped.split()) <= 6
        and re.search(r"[A-Za-z]-\s+[A-Za-z]", stripped)
    ):
        return True
    if stripped.count("\n") >= 3 and max((len(line.strip()) for line in stripped.splitlines()), default=0) <= 3:
        return True
    return False


def boundary_quality_issue(label: str, text: str) -> str | None:
    stripped = " ".join(text.split()).strip()
    lowered = stripped.lower()
    if not stripped:
        return "whitespace_only"
    if stripped.startswith((")", "]", "}", "”", "’", '"', "'", ",", ";", ":")):
        return "starts_with_closing_punctuation"
    if (
        re.match(r"^(?:ure|ation|tion|ment|ness|ing|ed|ly)-", lowered)
        and len(stripped.split()) <= 4
    ):
        return "possible_midword_start"
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", lowered)
    if words and words[-1] in INCOMPLETE_TRAILING_WORDS:
        return "incomplete_trailing_word"
    if re.fullmatch(r"[A-Za-z]{2,5}", stripped):
        if label == "StatisticalAnalysis" and lowered not in STATISTICAL_METHOD_CUES:
            return "short_statistical_fragment"
    if label == "StatisticalAnalysis":
        has_method_cue = any(cue in lowered for cue in STATISTICAL_METHOD_CUES)
        if len(words) < 3 and not has_method_cue:
            return "insufficient_statistical_evidence"
    return None



def normalize_candidate_span(
    *,
    source_text: str,
    label: str,
    text: str,
    start: int,
    end: int,
) -> Tuple[str, int, int, Dict[str, Any]]:
    """Return deterministic label-aware boundary repairs before registry validation."""
    original = {"text": text, "start": start, "end": end}
    if start < 0 or end > len(source_text) or end <= start:
        return text, start, end, {"changed": False, "reason": "invalid_offsets"}
    left, right = trim_outer_boundary(source_text, start, end)
    if label == "SampleSize":
        left, right = sample_size_boundary(source_text, left, right)
    elif label == "ComparisonGroup":
        left, right = cue_phrase_boundary(
            source_text,
            left,
            right,
            r"(?i)\b(?:business[-\s]as[-\s]usual|BAU|wait[-\s]?list(?:\s+control)?|usual\s+care|control\s+group|placebo|no\s+intervention|active\s+comparator)\b[^.;,)]*",
        )
    elif label == "DesignDescription":
        left, right = cue_phrase_boundary(
            source_text,
            left,
            right,
            r"(?i)\b(?:cluster\s+)?(?:randomi[sz]ed\s+controlled\s+trial|randomi[sz]ed\s+trial|quasi[-\s]experimental|pretest[-\s]posttest|cross[-\s]sectional|cohort|factorial\s+design|mixed\s+factorial)\b[^.;]*",
        )
    elif label == "StatisticalAnalysis":
        left, right = trim_analysis_boundary(source_text, left, right)
    else:
        left, right = trim_leading_intro_words(source_text, left, right)
    normalized_text = source_text[left:right]
    changed = (left, right, normalized_text) != (start, end, text)
    metadata: Dict[str, Any] = {"changed": changed}
    if changed:
        metadata.update({"original": original, "normalized": {"text": normalized_text, "start": left, "end": right}})
    return normalized_text, left, right, metadata


def trim_outer_boundary(source_text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and source_text[start].isspace():
        start += 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    while start < end and source_text[start] in ",;:)]}”’\"'":
        start += 1
    while end > start and source_text[end - 1] in ",;:([{“‘\"'":
        end -= 1
    return start, end


def sample_size_boundary(source_text: str, start: int, end: int) -> Tuple[int, int]:
    segment = source_text[start:end]
    unit = r"(?:participants?|students?|teachers?|children|adolescents?|schools?|classrooms?|parents?|families|subjects?|pupils|youths?|dyads?|clusters?|cases?)"
    patterns = [
        rf"(?i)\b(?:intervention|control|comparison|treatment|wait[-\s]?list|usual\s+care)?\s*n\s*=\s*\d[\d,]*(?:\.\d+)?(?:\s*{unit})?",
        rf"(?i)\bN\s*=\s*\d[\d,]*(?:\.\d+)?(?:\s*{unit})?",
        rf"(?i)\b\d[\d,]*(?:\.\d+)?\s+{unit}\b",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, segment))
    if not matches:
        return start, end
    best = max(matches, key=lambda match: (len(match.group(0)), -match.start()))
    return start + best.start(), start + best.end()


def cue_phrase_boundary(source_text: str, start: int, end: int, pattern: str) -> Tuple[int, int]:
    segment = source_text[start:end]
    matches = list(re.finditer(pattern, segment))
    if not matches:
        return start, end
    best = max(matches, key=lambda match: (len(match.group(0)), -match.start()))
    left, right = start + best.start(), start + best.end()
    return trim_outer_boundary(source_text, left, right)


def trim_analysis_boundary(source_text: str, start: int, end: int) -> Tuple[int, int]:
    method_pattern = r"(?i)\b(?:multilevel\s+linear\s+model(?:ing)?|linear\s+mixed\s+model(?:s)?|mixed[-\s]effects?\s+model(?:s)?|ANCOVA|ANOVA|logistic\s+regression|linear\s+regression|t[-\s]?test|chi[-\s]?square|intention[-\s]to[-\s]treat|last\s+observation\s+carried\s+forward|Hedges'?\s+g|Cohen'?s\s+d|odds\s+ratio|confidence\s+intervals?)\b[^.;]*"
    return cue_phrase_boundary(source_text, start, end, method_pattern)


def trim_leading_intro_words(source_text: str, start: int, end: int) -> Tuple[int, int]:
    segment = source_text[start:end]
    match = re.match(
        r"(?i)^(?:the\s+)?(?:study\s+)?(?:included|recruited|enrolled|assessed|measured|reported|received|were|was|are|is)\s+",
        segment,
    )
    if match:
        start += match.end()
    return trim_outer_boundary(source_text, start, end)


def validate_candidate_text(
    *,
    source_text: str,
    label: str,
    text: str,
    start: int,
    end: int,
) -> CleaningDecision:
    if label not in VALID_LABELS:
        return CleaningDecision(False, "unsupported_label", "error")
    if start < 0 or end <= start or end > len(source_text):
        return CleaningDecision(False, "invalid_offsets", "error")
    if source_text[start:end] != text:
        return CleaningDecision(False, "text_offset_mismatch", "error")
    if not text.strip():
        return CleaningDecision(False, "whitespace_only", "error")
    if is_punctuation_only(text):
        return CleaningDecision(False, "punctuation_only", "error")
    if is_isolated_stopword(text):
        return CleaningDecision(False, "isolated_stopword", "error")
    if is_one_character_fragment(label, text):
        return CleaningDecision(False, "one_character_alphabetic_fragment", "error")
    if looks_like_broken_subword(text):
        return CleaningDecision(False, "broken_subword_fragment", "warning")
    if looks_like_malformed_pdf_fragment(text):
        return CleaningDecision(False, "malformed_pdf_fragment", "warning")
    if label == "SampleSize" and re.match(r"^\(\s*[Nn]\s*=", text.strip()) and not text.strip().endswith(")"):
        return CleaningDecision(False, "incomplete_sample_size_parenthesis", "warning")
    boundary_issue = boundary_quality_issue(label, text)
    if boundary_issue:
        return CleaningDecision(False, boundary_issue, "warning")
    return CleaningDecision(True, "accepted")
