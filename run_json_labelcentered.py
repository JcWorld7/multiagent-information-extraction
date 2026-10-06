"""Run label-centered PICO extraction on Label Studio JSON records."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.consultation import run_labelcentered_consultation
from Labelcentered.expert_models import ExpertEnsemble, raw_candidates_to_json, thresholds_by_expert_and_label
from Labelcentered.json_input import build_model_input_from_record, deduplicate_paths, load_records, record_name, safe_name
from Labelcentered.llm_client import LLMClient
from Labelcentered.output_validation import validate_output
from Labelcentered.settings import (
    DEFAULT_DATASET_PATHS,
    DEFAULT_LLM_BACKEND,
    LLM_BACKENDS,
    FINAL_SELECTION_POLICIES,
    PipelineSettings,
    llm_backend_metadata,
    llm_backend_output_dir,
    llm_backend_spec,
)


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


def process_record(record: Dict[str, Any], json_path: Path, index: int, out_dir: Path, settings: PipelineSettings, llm_client: LLMClient, classifier_only: bool, skip_existing: bool) -> str:
    rec_name = record_name(record, index)
    output_path = out_dir / f"{safe_name(json_path.stem)}__{rec_name}.labelcentered.json"
    if skip_existing and output_path.exists():
        print(f"[skip] {rec_name}")
        return "skipped"
    source_text = build_model_input_from_record(record)
    if not source_text.strip():
        raise ValueError("Record has empty method/results model input")
    ensemble = ExpertEnsemble(settings)
    raw = ensemble.extract_by_expert_and_label(source_text)
    result = {
        "source_file": json_path.name,
        "source_path": str(json_path),
        "record_index": index,
        "record_name": rec_name,
        "input_type": "json",
        "model_input_text": source_text,
        "llm_backend": settings.llm_backend,
        "llm_model": settings.llm_model,
        "llm_agent_backend": llm_backend_metadata(settings),
        "expert_models": expert_spec_json(settings),
        "thresholds_by_expert_and_label": thresholds_by_expert_and_label(settings),
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
    parser = argparse.ArgumentParser(description="Run label-centered PICO extraction on JSON records.")
    parser.add_argument("--json", nargs="*")
    parser.add_argument("--out_dir", help="Defaults to the selected backend output folder, e.g. outputs/qwen")
    parser.add_argument("--limit_records", type=int)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--classifier_only", action="store_true")
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
    paths = [Path(x) for x in args.json] if args.json else list(DEFAULT_DATASET_PATHS)
    paths = [path for path in deduplicate_paths(paths) if path.exists()]
    if not paths:
        raise FileNotFoundError("No JSON files found")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else llm_backend_output_dir(args.llm_backend).resolve()
    llm_client = LLMClient(settings)
    if not args.classifier_only and not args.skip_llm_model_check:
        llm_client.validate_model_available()
    totals = {"processed": 0, "skipped": 0, "failed": 0}
    for path in paths:
        records = load_records(path)
        if args.limit_records is not None:
            records = records[: args.limit_records]
        for index, record in enumerate(records):
            try:
                status = process_record(record, path, index, out_dir, settings, llm_client, args.classifier_only, args.skip_existing)
                totals[status] += 1
            except Exception as exc:
                print(f"[error] {path} record {index}: {exc}")
                totals["failed"] += 1
    print(totals)


if __name__ == "__main__":
    main()
