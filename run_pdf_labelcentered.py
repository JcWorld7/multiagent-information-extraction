"""Run label-centered PICO extraction on PDFs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.consultation import run_labelcentered_consultation
from Labelcentered.expert_models import ExpertEnsemble, raw_candidates_to_json, thresholds_by_expert_and_label
from Labelcentered.json_input import deduplicate_paths, safe_name
from Labelcentered.llm_client import LLMClient
from Labelcentered.pdf_input import build_model_input_from_pdf, extract_methods_results
from Labelcentered.settings import (
    DEFAULT_LLM_BACKEND,
    LLM_BACKENDS,
    FINAL_SELECTION_POLICIES,
    PipelineSettings,
    llm_backend_metadata,
    llm_backend_output_dir,
    llm_backend_spec,
)
from Labelcentered.output_validation import validate_output


def expert_spec_json(settings: PipelineSettings) -> List[Dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "base_encoder": spec.base_encoder,
            "checkpoint_dir": str(spec.checkpoint_dir),
            "thresholds_path": str(spec.resolved_thresholds_path()),
        }
        for spec in settings.experts
    ]


def process_pdf(pdf_path: Path, out_dir: Path, settings: PipelineSettings, llm_client: LLMClient, classifier_only: bool, save_text: bool, skip_existing: bool) -> str:
    output_path = out_dir / f"{safe_name(pdf_path.stem)}.labelcentered.json"
    if skip_existing and output_path.exists():
        print(f"[skip] {pdf_path.name}")
        return "skipped"
    sections = extract_methods_results(pdf_path)
    if not sections.get("method") and not sections.get("results"):
        raise ValueError("No Methods or Results text found in PDF")
    source_text = build_model_input_from_pdf(pdf_path)
    ensemble = ExpertEnsemble(settings)
    raw = ensemble.extract_by_expert_and_label(source_text)
    thresholds = thresholds_by_expert_and_label(settings)
    result: Dict[str, Any] = {
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "input_type": "pdf",
        "model_input_text": source_text,
        "llm_backend": settings.llm_backend,
        "llm_model": settings.llm_model,
        "llm_agent_backend": llm_backend_metadata(settings),
        "expert_models": expert_spec_json(settings),
        "thresholds_by_expert_and_label": thresholds,
        "raw_candidates_by_expert_and_label": raw_candidates_to_json(raw),
        **run_labelcentered_consultation(
            source_text=source_text,
            raw_candidates_by_expert_and_label=raw,
            settings=settings,
            llm_client=llm_client,
            classifier_only=classifier_only,
        ),
    }
    errors = validate_output(result)
    if errors:
        raise ValueError(f"Output validation failed: {errors}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved -> {output_path}")
    return "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-centered PICO extraction on PDFs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", nargs="+")
    group.add_argument("--pdf_dir")
    parser.add_argument("--out_dir", help="Defaults to the selected backend output folder, e.g. outputs/qwen")
    parser.add_argument("--classifier_only", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--save_text", action="store_true")
    parser.add_argument("--generate_summaries", action="store_true")
    parser.add_argument("--allow_default_thresholds", action="store_true")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--consensus_rule", default="unanimous")
    parser.add_argument("--final_selection_policy", choices=FINAL_SELECTION_POLICIES, default="expert_consensus_verified")
    parser.add_argument("--calibration_config", help="JSON config from tune_labelcentered_calibration.py; required for --final_selection_policy calibrated")
    parser.add_argument("--doctor_batch_size", type=int, default=25)
    parser.add_argument("--llm_backend", choices=sorted(LLM_BACKENDS), default=DEFAULT_LLM_BACKEND)
    parser.add_argument("--llm_model", help="Override the model name for the selected backend")
    parser.add_argument("--llm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm_api_key", default="EMPTY")
    parser.add_argument("--llm_temperature", type=float, default=0.2)
    parser.add_argument("--llm_max_tokens", type=int, default=4096)
    parser.add_argument("--llm_timeout_seconds", type=float, default=900.0)
    parser.add_argument("--llm_context_window", type=int)
    parser.add_argument("--skip_llm_model_check", action="store_true")
    parser.add_argument("--normalize_candidate_boundaries", action="store_true")
    parser.add_argument("--disable_deterministic_pre_expansion", action="store_true")
    args = parser.parse_args()
    backend = llm_backend_spec(args.llm_backend)
    settings = PipelineSettings(
        allow_default_thresholds=args.allow_default_thresholds,
        max_rounds=args.max_rounds,
        consensus_rule=args.consensus_rule,
        doctor_batch_size=args.doctor_batch_size,
        generate_summaries=args.generate_summaries,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model or backend.model,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.llm_temperature,
        llm_max_tokens=args.llm_max_tokens,
        llm_context_window=args.llm_context_window or backend.context_window,
        llm_timeout_seconds=args.llm_timeout_seconds,
        calibration_config_path=args.calibration_config,
    )
    object.__setattr__(settings, "final_selection_policy", args.final_selection_policy)
    object.__setattr__(settings, "normalize_candidate_boundaries", args.normalize_candidate_boundaries)
    object.__setattr__(settings, "deterministic_pre_expansion", not args.disable_deterministic_pre_expansion)
    paths = [Path(x) for x in args.pdf] if args.pdf else sorted(Path(args.pdf_dir).expanduser().glob("*.pdf"))
    paths = deduplicate_paths(paths)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else llm_backend_output_dir(args.llm_backend).resolve()
    llm_client = LLMClient(settings)
    if not args.classifier_only and not args.skip_llm_model_check:
        llm_client.validate_model_available()
    totals = {"processed": 0, "skipped": 0, "failed": 0}
    for path in paths:
        try:
            status = process_pdf(path, out_dir, settings, llm_client, args.classifier_only, args.save_text, args.skip_existing)
            totals[status] += 1
        except Exception as exc:
            print(f"[error] {path}: {exc}")
            totals["failed"] += 1
    print(totals)


if __name__ == "__main__":
    main()
