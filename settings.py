"""Portable settings for the label-centered PICO pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


LABELS: List[str] = [
    "Participant",
    "Intervention",
    "Outcome",
    "SampleSize",
    "ComparisonGroup",
    "DesignDescription",
    "StatisticalAnalysis",
]
TYPE2ID: Dict[str, int] = {label: index for index, label in enumerate(LABELS)}
ID2TYPE: Dict[int, str] = {index: label for label, index in TYPE2ID.items()}

LABEL_PREFIX = {
    "Participant": "PAR",
    "Intervention": "INT",
    "Outcome": "OUT",
    "SampleSize": "SAM",
    "ComparisonGroup": "COM",
    "DesignDescription": "DES",
    "StatisticalAnalysis": "STA",
}

DOCTOR_NAMES = ["SciBERT", "PubMedBERT", "ClinicalBERT"]
FINAL_SELECTION_POLICIES: List[str] = [
    "strict",
    "recommended",
    "pubmedbert_only",
    "majority_vote",
    "pubmedbert_anchor",
    "pubmedbert_plus_verified",
    "expert_consensus_verified",
    "calibrated",
    "registry_filtered",
]
DOCTOR_ROLE_TEMPLATE = "{expert}_{label}_Doctor"
META_ROLE_TEMPLATE = "{label}MetaAgent"

DEFAULT_IDAS_ROOT = Path("/home/zhf/hpchome/ModelForPICO/ExtractiveModel")
DEFAULT_AGENT_ROOT = DEFAULT_IDAS_ROOT / "Agent"
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass(frozen=True)
class LLMBackendSpec:
    key: str
    model: str
    output_dir: Path
    context_window: int = 16384


LLM_BACKENDS: Dict[str, LLMBackendSpec] = {
    "qwen": LLMBackendSpec(
        key="qwen",
        model="Qwen/Qwen2.5-7B-Instruct",
        output_dir=DEFAULT_OUTPUT_ROOT / "qwen",
    ),
    "llama31_8b": LLMBackendSpec(
        key="llama31_8b",
        model="meta-llama/Llama-3.1-8B-Instruct",
        output_dir=DEFAULT_OUTPUT_ROOT / "llama31_8b",
    ),
    "mistral7b": LLMBackendSpec(
        key="mistral7b",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        output_dir=DEFAULT_OUTPUT_ROOT / "mistral7b",
    ),
}
DEFAULT_LLM_BACKEND = "qwen"


def llm_backend_spec(backend: str) -> LLMBackendSpec:
    try:
        return LLM_BACKENDS[backend]
    except KeyError as exc:
        valid = ", ".join(sorted(LLM_BACKENDS))
        raise ValueError(f"Unknown LLM backend {backend!r}; choose one of: {valid}") from exc


def llm_backend_output_dir(backend: str) -> Path:
    return llm_backend_spec(backend).output_dir


def llm_backend_metadata(settings: "PipelineSettings") -> Dict[str, str | float | int]:
    return {
        "backend": settings.llm_backend,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "context_window": settings.llm_context_window,
        "timeout_seconds": settings.llm_timeout_seconds,
    }


@dataclass(frozen=True)
class ExpertSpec:
    name: str
    base_encoder: str
    checkpoint_dir: Path
    thresholds_path: Path | None = None

    def resolved_thresholds_path(self) -> Path:
        return self.thresholds_path or (self.checkpoint_dir / "best_thresholds.json")


def default_experts(root: Path = DEFAULT_IDAS_ROOT) -> List[ExpertSpec]:
    return [
        ExpertSpec(
            name="SciBERT",
            base_encoder="allenai/scibert_scivocab_uncased",
            checkpoint_dir=root / "10times250filesSciBert/eight/checkpoint-1833",
        ),
        ExpertSpec(
            name="PubMedBERT",
            base_encoder="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
            checkpoint_dir=root / "10times250filesPubMed/eight/checkpoint-2145",
        ),
        ExpertSpec(
            name="ClinicalBERT",
            base_encoder="emilyalsentzer/Bio_ClinicalBERT",
            checkpoint_dir=root / "10times250filesClinical/eight/checkpoint-1043",
        ),
    ]


DEFAULT_DATASET_PATHS = [
    DEFAULT_IDAS_ROOT / "Data/10_Extraction_091425.json",
    DEFAULT_IDAS_ROOT / "Data/20_Extraction_091425.json",
    DEFAULT_IDAS_ROOT / "Data/30_Extraction_091425.json",
    DEFAULT_IDAS_ROOT / "Data/40_Extraction_091425.json",
    DEFAULT_IDAS_ROOT / "Data/50_Extraction_091425.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_1_10_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_11_20_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_21_30_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_31_40_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_41_50_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_51_60_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_61_70_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_71_80_Nov.8.json",
    DEFAULT_IDAS_ROOT / "Data/first10_060425.json",
    DEFAULT_IDAS_ROOT / "Data/first20_060425.json",
    DEFAULT_IDAS_ROOT / "Data/first30_060425.json",
    DEFAULT_IDAS_ROOT / "Data/first40_060425.json",
    DEFAULT_IDAS_ROOT / "Data/first50_060425.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_81_90_Nov.31_n11.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_91_100_Nov.31_n11.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_101_110_Nov.31.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_111_120_Dec.30.json",
    DEFAULT_IDAS_ROOT / "Data/Ex_121_130_Jan.15_2026.json",
    DEFAULT_IDAS_ROOT / "Data/10_Ext_01_26.json",
    DEFAULT_IDAS_ROOT / "Data/20_Ext_01_26.json",
]


@dataclass
class PipelineSettings:
    experts: List[ExpertSpec] = field(default_factory=default_experts)
    dataset_paths: List[Path] = field(default_factory=lambda: list(DEFAULT_DATASET_PATHS))
    llm_backend: str = DEFAULT_LLM_BACKEND
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_model: str = LLM_BACKENDS[DEFAULT_LLM_BACKEND].model
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    llm_context_window: int = 16384
    llm_context_reserve_tokens: int = 128
    llm_min_output_tokens: int = 256
    llm_timeout_seconds: float = 900.0
    max_rounds: int = 3
    consensus_rule: str = "unanimous"
    final_selection_policy: str = "expert_consensus_verified"
    doctor_batch_size: int = 25
    retry_limit: int = 3
    max_len: int = 512
    stride: int = 128
    infer_batch: int = 8
    allow_default_thresholds: bool = False
    generate_summaries: bool = False
    normalize_candidate_boundaries: bool = False
    calibration_config_path: str | None = None
    deterministic_pre_expansion: bool = True
    max_doctor_expansion_requests_per_label_round: int = 6


def role_name(expert: str, label: str) -> str:
    return DOCTOR_ROLE_TEMPLATE.format(expert=expert, label=label)


def meta_role_name(label: str) -> str:
    return META_ROLE_TEMPLATE.format(label=label)
