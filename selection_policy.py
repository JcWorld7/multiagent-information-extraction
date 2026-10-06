"""Deterministic candidate selection policies for ablation and final assembly."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


STRICT_LABELS = {"Outcome", "SampleSize", "ComparisonGroup"}


def candidate_id(candidate: Any) -> str:
    return str(get_value(candidate, "candidate_id"))


def candidate_label(candidate: Any) -> str:
    return str(get_value(candidate, "label"))


def candidate_text(candidate: Any) -> str:
    return str(get_value(candidate, "text"))


def candidate_start(candidate: Any) -> int:
    return int(get_value(candidate, "start"))


def candidate_end(candidate: Any) -> int:
    return int(get_value(candidate, "end"))


def candidate_experts(candidate: Any) -> list[str]:
    value = get_value(candidate, "proposed_by_experts", [])
    return [str(item) for item in value or []]


def get_value(candidate: Any, key: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def proposed_by_pubmedbert(candidate: Any) -> bool:
    return any(expert.lower() == "pubmedbert" for expert in candidate_experts(candidate))


def expert_support_count(candidate: Any) -> int:
    return len({expert.lower() for expert in candidate_experts(candidate)})


def keep_vote_count(vote_matrix: Mapping[str, Mapping[str, str]], cid: str) -> int:
    return sum(vote == "keep" for vote in vote_matrix.get(cid, {}).values())


def passes_label_filter(candidate: Any) -> bool:
    label = candidate_label(candidate)
    text = clean_text(candidate_text(candidate))
    lowered = text.lower()
    if not text:
        return False
    if label == "SampleSize":
        return sample_size_filter(text, lowered)
    if label == "Outcome":
        return outcome_filter(text, lowered)
    if label == "ComparisonGroup":
        return comparison_filter(text, lowered)
    if label == "StatisticalAnalysis":
        return statistical_filter(text, lowered)
    if label == "DesignDescription":
        return design_filter(text, lowered)
    if label == "Participant":
        return participant_filter(text, lowered)
    if label == "Intervention":
        return intervention_filter(text, lowered)
    return True


def sample_size_filter(text: str, lowered: str) -> bool:
    if not re.search(r"\d", text):
        return False
    statistical_value = (
        r"\bp\s*[<=>]|\bf\s*\(|\bχ2\b|\bchi[-\s]?square\b|\bβ\b|\bbeta\b|"
        r"\bci\b|\bodds\s+ratio\b|\bor\s*[=:]|\bsd\b|\bm\s*=|"
        r"\bmean\b|\bage\b|\byears?\s+old\b"
    )
    if re.search(statistical_value, lowered):
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?%", text.strip()):
        return False
    unit = r"participants?|students?|teachers?|children|adolescents?|schools?|classrooms?|parents?|families|subjects?|pupils|youths?|dyads?|clusters?|kindergartens?"
    group = r"intervention|control|comparison|treatment|wait[-\s]?list|experimental"
    return bool(
        re.search(rf"\b[Nn]\s*=\s*\d[\d,]*(?:\.\d+)?(?:\s*{unit})?\b", text)
        or re.search(rf"\b\d[\d,]*(?:\.\d+)?\s+{unit}\b", lowered)
        or re.search(rf"\b(?:{group})\s+n\s*=\s*\d[\d,]*(?:\.\d+)?\b", lowered)
        or re.search(rf"\b\d[\d,]*(?:\.\d+)?\s+(?:from|in|assigned to|allocated to)\s+(?:the\s+)?(?:{group})\b", lowered)
        or re.search(rf"\b\d[\d,]*(?:\.\d+)?.*\b(?:{group})\b.*\b\d[\d,]*(?:\.\d+)?\b", lowered)
    )


def outcome_filter(text: str, lowered: str) -> bool:
    hard_reject = [
        r"\b(?:anova|ancova|regression|mixed\s+model|t[-\s]?test|chi[-\s]?square|p\s*[<=>]|cohen|hedges|odds\s+ratio)\b",
        r"\b(?:intervention|program|curriculum|lesson|training|therapy|treatment)\b.*\b(?:delivered|session|weekly)\b",
        r"\b(?:instruments?|measures?)\s+were\s+used\b",
        r"\b(?:sociodemographic|demographic)\s+questionnaire\b",
    ]
    if any(re.search(pattern, lowered) for pattern in hard_reject):
        return False
    cues = [
        "outcome", "measure", "scale", "score", "symptom", "depression", "depressive", "anxiety",
        "stress", "well-being", "wellbeing", "self-esteem", "behavior", "behaviour", "prosocial",
        "conduct", "emotional", "mental health", "coping", "resilience", "competence", "adjustment",
        "inventory", "questionnaire", "instrument", "assessment", "index", "mood", "feelings",
        "beck depression", "bdi", "ces-d", "sdq",
    ]
    return any(cue in lowered for cue in cues)


def comparison_filter(text: str, lowered: str) -> bool:
    if lowered.strip(" .,:;()[]{}") in {"control", "wait", "drop", "comparison"}:
        return False
    cues = [
        "control", "wait-list", "waitlist", "usual care", "business-as-usual", "bau", "placebo",
        "comparison group", "no intervention", "active comparator", "normal activities",
    ]
    return any(cue in lowered for cue in cues)


def statistical_filter(text: str, lowered: str) -> bool:
    cues = [
        "anova", "ancova", "regression", "mixed model", "mixed-model", "multilevel", "t-test", "t test",
        "chi-square", "χ2", "imputation", "intention-to-treat", "cohen", "hedges", "odds ratio",
        "confidence interval", "covariate", "bonferroni", "spss", "stata", "effect size",
    ]
    return any(cue in lowered for cue in cues)


def design_filter(text: str, lowered: str) -> bool:
    cues = [
        "randomized", "randomised", "controlled trial", "rct", "quasi-experimental", "experiment",
        "pretest", "posttest", "follow-up", "crossover", "cross-sectional", "cohort", "factorial",
        "randomization", "randomisation", "assigned", "design",
    ]
    return any(cue in lowered for cue in cues)


def participant_filter(text: str, lowered: str) -> bool:
    cues = ["student", "teacher", "child", "children", "adolescent", "participant", "parent", "classroom", "school", "kindergarten"]
    return any(cue in lowered for cue in cues)


def intervention_filter(text: str, lowered: str) -> bool:
    if comparison_filter(text, lowered) and not re.search(r"\b(?:intervention|program|curriculum|therapy|training|lesson|counseling|mindfulness|cbt|sel)\b", lowered):
        return False
    cues = ["intervention", "program", "programme", "curriculum", "therapy", "training", "lesson", "counseling", "mindfulness", "cbt", "sel", "sessions"]
    return any(cue in lowered for cue in cues)


def select_pubmedbert_plus_verified(
    candidates: Sequence[Any],
    vote_matrix: Mapping[str, Mapping[str, str]],
) -> list[str]:
    eligible = [candidate for candidate in candidates if passes_label_filter(candidate)]
    kept: list[Any] = []
    for candidate in eligible:
        if proposed_by_pubmedbert(candidate) and not weak_fragment(candidate):
            kept.append(candidate)
    for candidate in eligible:
        cid = candidate_id(candidate)
        if any(candidate_id(existing) == cid for existing in kept):
            continue
        keep_count = keep_vote_count(vote_matrix, cid)
        if proposed_by_pubmedbert(candidate):
            if weak_fragment(candidate):
                continue
            kept.append(candidate)
            continue
        if keep_count >= 2 or high_quality_rescue(candidate, candidates, vote_matrix):
            kept.append(candidate)
    return [candidate_id(candidate) for candidate in prune_redundant(kept)]


def select_expert_consensus_verified(
    candidates: Sequence[Any],
    vote_matrix: Mapping[str, Mapping[str, str]],
) -> list[str]:
    """Equal-expert final policy tuned for token-level precision/recall balance.

    No encoder gets anchor status. Multi-expert agreement is accepted as the
    strongest model-side signal; single-expert or expanded candidates need
    doctor support or a narrow deterministic rescue rule.
    """
    eligible = [candidate for candidate in candidates if passes_label_filter(candidate)]
    kept: list[Any] = []
    for candidate in eligible:
        if weak_fragment(candidate):
            continue
        keep_count = keep_vote_count(vote_matrix, candidate_id(candidate))
        if expert_support_count(candidate) >= 2:
            kept.append(candidate)
            continue
        if keep_count >= 2 or high_quality_rescue(candidate, candidates, vote_matrix):
            kept.append(candidate)
    return [candidate_id(candidate) for candidate in prune_redundant(kept)]


def high_quality_rescue(
    candidate: Any,
    candidates: Sequence[Any],
    vote_matrix: Mapping[str, Mapping[str, str]],
) -> bool:
    label = candidate_label(candidate)
    if label not in STRICT_LABELS:
        return False
    if weak_fragment(candidate):
        return False
    if keep_vote_count(vote_matrix, candidate_id(candidate)) >= 1:
        return True
    if label == "SampleSize":
        return sample_size_quality(candidate) >= 2
    if label == "ComparisonGroup":
        return comparison_quality(candidate) >= 2 and word_count(candidate_text(candidate)) <= 8
    if label == "Outcome":
        return outcome_quality(candidate) >= 2 and word_count(candidate_text(candidate)) <= 10
    return expands_any_weak_signal(candidate, candidates)


def prune_redundant(candidates: Sequence[Any]) -> list[Any]:
    kept: list[Any] = []
    for candidate in candidates:
        duplicate_index = next(
            (index for index, existing in enumerate(kept) if near_duplicate(candidate, existing)),
            None,
        )
        if duplicate_index is None:
            kept.append(candidate)
            continue
        existing = kept[duplicate_index]
        if better_boundary(candidate, existing):
            kept[duplicate_index] = candidate
    return kept


def near_duplicate(left: Any, right: Any) -> bool:
    if candidate_label(left) != candidate_label(right):
        return False
    overlap = max(0, min(candidate_end(left), candidate_end(right)) - max(candidate_start(left), candidate_start(right)))
    shorter = max(1, min(candidate_end(left) - candidate_start(left), candidate_end(right) - candidate_start(right)))
    if overlap / shorter >= 0.85:
        return True
    left_text = normalized_text(candidate_text(left))
    right_text = normalized_text(candidate_text(right))
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if candidate_label(left) in STRICT_LABELS:
        return left_text in right_text or right_text in left_text
    return False


def better_boundary(candidate: Any, existing: Any) -> bool:
    candidate_score = boundary_score(candidate)
    existing_score = boundary_score(existing)
    if weak_fragment(existing) and not weak_fragment(candidate):
        return True
    if proposed_by_pubmedbert(existing) and not proposed_by_pubmedbert(candidate):
        return candidate_score > existing_score
    if proposed_by_pubmedbert(candidate) and not proposed_by_pubmedbert(existing):
        return candidate_score >= existing_score
    return candidate_score > existing_score


def boundary_score(candidate: Any) -> tuple[int, int, int, int]:
    label = candidate_label(candidate)
    text = clean_text(candidate_text(candidate))
    score = 0
    if passes_label_filter(candidate):
        score += 10
    if text and text[0] not in ",;:.)]}" and text[-1] not in ",;:([{":
        score += 2
    if weak_fragment(candidate):
        score -= 8
    if label == "SampleSize":
        score += sample_size_quality(candidate)
    elif label == "Outcome":
        score += outcome_quality(candidate)
    elif label == "ComparisonGroup":
        score += comparison_quality(candidate)
    if label in STRICT_LABELS:
        words = word_count(text)
        if words > 14:
            score -= words - 14
        if 2 <= words <= 8:
            score += 2
    return score, -abs(word_count(text) - preferred_word_count(label)), -len(text), -candidate_start(candidate)


def sample_size_quality(candidate: Any) -> int:
    lowered = clean_text(candidate_text(candidate)).lower()
    score = 0
    if re.search(r"\d", lowered):
        score += 1
    if re.search(r"\b(?:participants?|students?|children|adolescents?|schools?|classrooms?|parents?|families|subjects?|pupils)\b", lowered):
        score += 1
    if re.search(r"\b(?:intervention|control|comparison|treatment|wait[-\s]?list|experimental)\b", lowered):
        score += 1
    if re.search(r"\b\d[\d,]*(?:\.\d+)?.*\b\d[\d,]*(?:\.\d+)?\b", lowered):
        score += 1
    return score


def outcome_quality(candidate: Any) -> int:
    lowered = clean_text(candidate_text(candidate)).lower()
    score = 0
    if re.search(r"\b(?:scale|score|inventory|questionnaire|instrument|assessment|index)\b", lowered):
        score += 2
    if re.search(r"\b(?:depression|depressive|anxiety|mood|feelings|symptoms?|well-?being|self-esteem|behavior|behaviour)\b", lowered):
        score += 1
    if word_count(lowered) >= 2:
        score += 1
    return score


def comparison_quality(candidate: Any) -> int:
    lowered = clean_text(candidate_text(candidate)).lower()
    score = 0
    if re.search(r"\b(?:wait[-\s]?list|usual care|business-as-usual|bau|placebo|no intervention|active comparator)\b", lowered):
        score += 2
    if re.search(r"\b(?:control|comparison)\b", lowered):
        score += 1
    if re.search(r"\b(?:group|school|classroom|condition)\b", lowered):
        score += 1
    if word_count(lowered) >= 2:
        score += 1
    return score


def weak_fragment(candidate: Any) -> bool:
    label = candidate_label(candidate)
    text = clean_text(candidate_text(candidate))
    lowered = normalized_text(text)
    words = word_count(text)
    if label == "Outcome":
        return words <= 1 and lowered in {
            "depression", "mood", "feelings", "satisfaction", "anxiety", "stress", "score", "outcome",
        }
    if label == "SampleSize":
        if not re.search(r"\d", text):
            return True
        return words == 1 and not proposed_by_pubmedbert(candidate)
    if label == "ComparisonGroup":
        return words <= 1 or lowered in {"control", "wait", "drop", "comparison"}
    return False


def expands_any_weak_signal(candidate: Any, candidates: Sequence[Any]) -> bool:
    text = normalized_text(candidate_text(candidate))
    if not text:
        return False
    for other in candidates:
        if candidate_id(other) == candidate_id(candidate):
            continue
        if candidate_label(other) != candidate_label(candidate):
            continue
        other_text = normalized_text(candidate_text(other))
        if other_text and other_text in text and weak_fragment(other):
            return True
    return False


def preferred_word_count(label: str) -> int:
    if label == "SampleSize":
        return 7
    if label == "Outcome":
        return 4
    if label == "ComparisonGroup":
        return 4
    return 6


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def normalized_text(text: str) -> str:
    return clean_text(text).lower().strip(" .,:;()[]{}")
