"""Annotation-guideline retrieval for label-centered extraction."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree

from Labelcentered.settings import LABELS


DEFAULT_GUIDELINE_DIR = Path(__file__).resolve().parent / "RAG_Documents"
LOCAL_SNIPPET_LIMIT_CHARS = 1400


GUIDELINES: Dict[str, Dict[str, List[str] | str]] = {
    "Participant": {
        "definition": (
            "Who was studied or received the school-based intervention, especially "
            "K-12 students and/or teachers, including grade, age, demographics, "
            "eligibility, risk status, and school context when tied to the sample."
        ),
        "include": [
            "population type and schooling level such as fifth-grade students, high school teachers, K-12 students",
            "eligibility or targeting such as below the 40th percentile, at risk for anxiety, depressive symptoms",
            "demographics or context tied to participants such as mean age, percent female, free/reduced lunch",
            "teacher, classroom, school, or student descriptors when they define the sample",
        ],
        "exclude": [
            "pure counts without descriptive sample context; those are SampleSize",
            "school or district names unless they define the participant group",
            "country or geography alone",
            "intervention recipients described only as arm labels",
        ],
        "boundary": [
            "Capture the smallest phrase that fully identifies the participant category.",
            "Trim recruitment verbs and surrounding methods text when a noun phrase is enough.",
            "Include grade, age, eligibility, or demographic modifiers when contiguous and informative.",
            "Multiple spans are allowed when Methods and Results each add new sample context.",
        ],
        "examples": [
            "Participants were 827 upper middle school students (after removing 10 pre-post dropouts) from 41 classes in five public school groupings in the Lisbon district",
            "Participants were 6,667 secondary school students from 40 state secondary schools in southeast England",
            "students scoring below the 40th percentile on a school-administered reading assessment",
            "teachers from three public high schools serving grades 9-12",
        ],
        "common_errors": [
            "Confusing sample counts with participant descriptions.",
            "Selecting a whole recruitment sentence when a precise noun phrase is available.",
        ],
    },
    "Intervention": {
        "definition": (
            "The active school-based mental health program, strategy, curriculum, "
            "activity, treatment, exposure, or support delivered to students and/or teachers."
        ),
        "include": [
            "program name or type such as mindfulness curriculum, CBT skills training, SEL program, Second Step",
            "delivery and dosage such as 12 weekly sessions, 30-minute classroom lessons, counselor-led group",
            "active components such as breathing exercises, cognitive restructuring, counseling, coping skills",
            "school-based prevention, promotion, wellness, bullying prevention, classroom guidance, or counseling lessons",
        ],
        "exclude": [
            "control, waitlist, usual-care, or comparator details; those are ComparisonGroup",
            "general study aims without delivered content",
            "outcome constructs that the intervention intends to improve",
        ],
        "boundary": [
            "Prefer the naming phrase plus essential modifiers for delivery, dose, and components.",
            "Keep comparator text out of Intervention spans even if both arms appear in one sentence.",
            "Do not expand to broad purpose or background language unless it is the only exact intervention description.",
        ],
        "examples": [
            "a 12-week mindfulness curriculum with 30-minute classroom sessions delivered weekly by trained teachers",
            "eight 45-minute CBT skills training sessions focused on cognitive restructuring and coping skills",
            "school counselor-led small-group counseling delivered during the regular school day",
            "the Second Step social-emotional learning program implemented by classroom teachers",
        ],
        "common_errors": [
            "Including the control arm.",
            "Selecting vague goal text instead of the delivered program or activity.",
        ],
    },
    "Outcome": {
        "definition": (
            "Measured mental health, social-emotional, behavioral, well-being, or related "
            "student/teacher endpoints, including constructs and instruments when they identify the measure."
        ),
        "include": [
            "mental health constructs such as anxiety, depression, stress, well-being, resilience, coping",
            "social-emotional skills, behavioral adjustment, prosocial behavior, bullying/safety, substance use",
            "observer-reported or self-reported checklist, inventory, score, scale, or symptom measure",
            "instrument plus construct when informative, such as RCADS total anxiety or teacher-reported SDQ Conduct Problems",
        ],
        "exclude": [
            "statistical models, tests, p-values, or effect-size reporting without an outcome construct",
            "instrument acronyms alone when the measured construct is available nearby",
            "phrases such as improved significantly when no measured construct is named",
        ],
        "boundary": [
            "Tag the measured construct and include the instrument only when it clarifies the construct.",
            "Use separate spans for separate constructs, such as anxiety symptoms and depression.",
            "Do not absorb analysis language around the outcome name.",
        ],
        "examples": [
            "student self-reported anxiety and depression symptoms measured with the RCADS",
            "teacher-reported SDQ Conduct Problems and Prosocial Behavior subscales",
            "perceived stress, resilience, and social-emotional well-being scores",
            "bullying victimization and school safety perceptions at post-test",
        ],
        "common_errors": [
            "Selecting regression or ANOVA text as Outcome.",
            "Selecting an acronym alone when the construct is present.",
        ],
    },
    "SampleSize": {
        "definition": (
            "Any numeric sample size for the whole study, analytic sample, arm, cluster, "
            "classroom, school, teacher group, or other unit."
        ),
        "include": [
            "overall N such as N=418 or 418 students",
            "arm counts such as intervention n=212 and control n=206",
            "cluster/unit counts such as 30 schools, 52 teachers, classrooms, families, dyads",
            "enrolled, allocated, randomized, completed, final analytic sample, and attrition-related counts",
        ],
        "exclude": [
            "percentages unless coupled with an N; tag the N-bearing phrase",
            "age ranges, grades, years, dates, p-values, confidence intervals, and effect sizes",
            "phrases like a large sample without a number",
        ],
        "boundary": [
            "Include the indicator and unit with the number, for example n=212 or 30 schools.",
            "Tag each arm or cluster count separately when reported separately.",
            "Short numeric spans are valid only when the surrounding phrase makes them sample counts.",
        ],
        "examples": [
            "N=418 fifth- and sixth-grade students from 20 public schools",
            "212 students in the intervention group and 206 students in the control group",
            "30 schools were randomized, with 15 assigned to the intervention and 15 to usual practice",
            "52 teachers and 1,184 students completed baseline surveys",
        ],
        "common_errors": [
            "Rejecting valid short numeric sample counts.",
            "Mistaking percentages, age, or statistical values for sample size.",
        ],
    },
    "ComparisonGroup": {
        "definition": (
            "The counterfactual or comparator arm: inactive control, waitlist, business-as-usual, "
            "usual care, placebo, baseline/no-intervention condition, or active alternative program."
        ),
        "include": [
            "business-as-usual (BAU), waitlist control, no intervention, usual care, placebo",
            "active comparator or alternative school-based mental health intervention",
            "brief comparator descriptors such as no intervention during the study period",
            "pre-post or baseline comparison phrasing only when it names the comparison condition",
        ],
        "exclude": [
            "intervention group details",
            "baseline participant characteristics",
            "statistical contrasts or results comparing arms",
        ],
        "boundary": [
            "Tag the comparator name or type, not the result sentence around it.",
            "If multiple comparators are named, tag each comparator phrase.",
            "Do not include claims that one group improved more than another.",
        ],
        "examples": [
            "business-as-usual classroom activities without additional mental health programming",
            "waitlist control group that received the program after the follow-up assessment",
            "active comparator: social skills training delivered by school counselors",
            "students in the control schools continued with usual health education lessons",
        ],
        "common_errors": [
            "Missing comparator phrases because they occur only in Methods.",
            "Including intervention-arm text in the comparator span.",
        ],
    },
    "DesignDescription": {
        "definition": (
            "Study design, assignment, timing, follow-up, and trial architecture, including "
            "randomized controlled, experimental, quasi-experimental, cohort, and pretest-posttest designs."
        ),
        "include": [
            "design type such as cluster randomized controlled trial, experiment, quasi-experimental, propensity score matching",
            "assignment level such as randomization at the school, classroom, student, or teacher level",
            "timing and follow-up descriptors such as pretest-posttest, 6-month follow-up, repeated time points",
            "factorial or mixed design descriptions when they define the study architecture",
        ],
        "exclude": [
            "statistical models and analytic procedures; those are StatisticalAnalysis",
            "implementation process narratives or intervention dosage alone",
            "general inclusion criteria for the review unless they describe the study design in the source text",
        ],
        "boundary": [
            "Tag the design phrase and key design modifiers.",
            "Timepoint statements can be separate spans if they define follow-up or design.",
            "Avoid selecting full Methods paragraphs unless the paragraph is primarily design description.",
        ],
        "examples": [
            "a cluster randomized controlled trial with schools as the unit of randomization",
            "a quasi-experimental pretest-posttest design with a matched comparison group",
            "outcomes were assessed immediately post-intervention and at 6-month follow-up",
            "the experimental design was a 2 x 2 mixed factorial design with group as a between-subjects factor and time as a within-subjects factor",
        ],
        "common_errors": [
            "Confusing intervention session structure with study design.",
            "Selecting multilevel models as DesignDescription instead of StatisticalAnalysis.",
        ],
    },
    "StatisticalAnalysis": {
        "definition": (
            "Analysis, modeling, testing, adjustment, missing-data handling, and effect-reporting "
            "methods used to evaluate outcomes."
        ),
        "include": [
            "models and tests such as multilevel linear modeling, ANCOVA, ANOVA, logistic regression, t-test",
            "effect metrics such as Hedges g, Cohen's d, odds ratio, beta coefficients, confidence intervals",
            "adjustments such as baseline covariates, clustering, ICC, school effects, multiple testing correction",
            "missing-data and analytic strategy terms such as intention-to-treat, last observation carried forward",
            "software or package names when they are part of the analytic method",
        ],
        "exclude": [
            "outcome names alone",
            "study design labels without analytic method details",
            "plain significance wording such as significant at p<.05 when no method/statistic is named",
        ],
        "boundary": [
            "Tag the model, test, metric, correction, or analytic clause.",
            "Adjacent values may be included when they define the statistic, for example OR=1.42.",
            "Broad spans are allowed when an entire sentence or paragraph is all about statistical analysis.",
        ],
        "examples": [
            "multilevel linear models adjusted for baseline scores and accounted for clustering of students within schools",
            "repeated measures ANOVAs examined intervention effects from pre-intervention to post-intervention",
            "intention-to-treat analysis was conducted using the last observation carried forward method for missing post-intervention data",
            "Hedges g effect sizes and 95% confidence intervals were reported for all primary outcomes",
        ],
        "common_errors": [
            "Confusing outcome constructs with analysis methods.",
            "Rejecting an analytic paragraph that should be selected as one broad StatisticalAnalysis span.",
        ],
    },
}


LABEL_HEADING_PATTERNS = {
    label: re.compile(rf"^(?:[A-G]\.\s*)?{re.escape(label)}\b", re.IGNORECASE)
    for label in LABELS
}

PICOS_LINE_MAP = {
    "population": "Participant",
    "interventions": "Intervention",
    "intervention": "Intervention",
    "comparison": "ComparisonGroup",
    "outcomes": "Outcome",
    "outcome": "Outcome",
}

PICOS_KEYWORD_MAP = {
    "Participant": ["K-12 students", "teachers from any cultural backgrounds"],
    "Intervention": ["school-based intervention", "Mental Health Promotion Programs", "Social-Emotional Learning"],
    "Outcome": ["Mental Health and Well-being", "Social-Emotional Skills", "Behavioral Adjustment", "Resilience and Coping"],
    "DesignDescription": ["randomized controlled trial", "experiment", "quasi-experimental"],
}


class GuidelineRetriever:
    def __init__(self, guideline_dir: Path | None = None):
        self.guideline_dir = guideline_dir if guideline_dir is not None else default_guideline_dir()
        self.local_docs = self._load_local_docs(self.guideline_dir) if self.guideline_dir else {}

    def retrieve(self, label: str) -> List[str]:
        if label not in LABELS:
            return []
        base = GUIDELINES[label]
        rows = [
            f"[{label}] definition: {base['definition']}",
            f"[{label}] include: {'; '.join(base['include'])}",
            f"[{label}] exclude: {'; '.join(base['exclude'])}",
            f"[{label}] boundary: {'; '.join(base['boundary'])}",
            f"[{label}] examples: {'; '.join(base['examples'])}",
            f"[{label}] example use: examples show acceptable complete spans, but final answers must copy only text present in the current source.",
            f"[{label}] common errors: {'; '.join(base['common_errors'])}",
        ]
        rows.extend(self.local_docs.get(label, []))
        return rows

    def _load_local_docs(self, guideline_dir: Path | None) -> Dict[str, List[str]]:
        docs: Dict[str, List[str]] = {label: [] for label in LABELS}
        if not guideline_dir or not guideline_dir.exists():
            return docs
        for path in sorted(guideline_dir.iterdir()):
            if path.suffix.lower() == ".txt":
                self._load_txt_doc(path, docs)
            elif path.suffix.lower() == ".docx":
                self._load_docx_doc(path, docs)
        return docs

    def _load_txt_doc(self, path: Path, docs: Dict[str, List[str]]) -> None:
        label = path.stem
        if label in docs:
            docs[label].append(format_local_snippet(path.name, path.read_text(encoding="utf-8")))

    def _load_docx_doc(self, path: Path, docs: Dict[str, List[str]]) -> None:
        paragraphs = docx_paragraphs(path)
        for label, text in label_snippets_from_paragraphs(paragraphs).items():
            if text:
                docs[label].append(format_local_snippet(path.name, text))


def default_guideline_dir() -> Path | None:
    return DEFAULT_GUIDELINE_DIR if DEFAULT_GUIDELINE_DIR.exists() else None


def docx_paragraphs(path: Path) -> List[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile):
        return []
    root = ElementTree.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        normalized = " ".join(text.split())
        if normalized:
            paragraphs.append(normalized)
    return paragraphs


def label_snippets_from_paragraphs(paragraphs: List[str]) -> Dict[str, str]:
    rows: Dict[str, List[str]] = {label: [] for label in LABELS}
    current_label: str | None = None
    for paragraph in paragraphs:
        heading_label = label_heading(paragraph)
        if heading_label:
            current_label = heading_label
            rows[current_label].append(paragraph)
            continue
        lower = paragraph.lower().strip()
        mapped = PICOS_LINE_MAP.get(lower.split(":", 1)[0].strip())
        if mapped:
            rows[mapped].append(paragraph)
        for label, keywords in PICOS_KEYWORD_MAP.items():
            if any(keyword.lower() in lower for keyword in keywords):
                rows[label].append(paragraph)
        if current_label and not starts_next_major_section(paragraph):
            rows[current_label].append(paragraph)
    return {
        label: trim_snippet(" ".join(deduplicate_preserve_order(rows[label])))
        for label in LABELS
    }


def label_heading(paragraph: str) -> str | None:
    stripped = paragraph.strip()
    for label, pattern in LABEL_HEADING_PATTERNS.items():
        if pattern.match(stripped):
            return label
    return None


def starts_next_major_section(paragraph: str) -> bool:
    return bool(re.match(r"^(?:[A-G]\.|General Annotation Rules|Goal:|Source:|Style:|PICO|Inclusion Criteria|Some Definitions)\b", paragraph))


def deduplicate_preserve_order(rows: List[str]) -> List[str]:
    output = []
    seen = set()
    for row in rows:
        if row not in seen:
            output.append(row)
            seen.add(row)
    return output


def trim_snippet(text: str, max_chars: int = LOCAL_SNIPPET_LIMIT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + " ...[truncated]"


def format_local_snippet(source_name: str, text: str) -> str:
    return f"[local guideline: {source_name}] {trim_snippet(text)}"
