"""Tune calibrated Labelcentered candidate selection from annotated data.

This script expects prediction payloads that already contain candidate
registries and doctor vote matrices. Generate those on IDAS first, then tune a
portable JSON config from gold Label Studio annotations.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.calibration import select_label_candidate_ids
from Labelcentered.evaluate_labelcentered import (
    EvalToken,
    EvalSpan,
    align_records,
    gold_spans_from_record,
    load_prediction_payloads,
    token_level_metrics,
    tokenize_with_offsets,
)
from Labelcentered.json_input import build_model_input_from_record, load_records, record_name
from Labelcentered.settings import DOCTOR_NAMES, LABELS


VOTE_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "equal": {expert: 1.0 for expert in DOCTOR_NAMES},
    "scibert_weighted": {"SciBERT": 1.25, "PubMedBERT": 1.0, "ClinicalBERT": 1.0},
    "pubmedbert_weighted": {"SciBERT": 1.0, "PubMedBERT": 1.25, "ClinicalBERT": 1.0},
    "clinicalbert_weighted": {"SciBERT": 1.0, "PubMedBERT": 1.0, "ClinicalBERT": 1.25},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Labelcentered calibrated selection config.")
    parser.add_argument("--gold", nargs="+", default=[], help="Gold Label Studio JSON files")
    parser.add_argument("--pred", nargs="*", default=[], help="Prediction .labelcentered.json files")
    parser.add_argument("--pred_dir", nargs="*", default=[], help="Prediction directories")
    parser.add_argument("--cache_json", nargs="*", default=[], help="Compact cache files from build_labelcentered_calibration_cache.py")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Optional named variant as NAME=prediction_dir, e.g. base=out normalized=out_norm",
    )
    parser.add_argument("--out_json", default="./labelcentered_calibration.json")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prefer", choices=["f1", "recall", "precision"], default="f1")
    parser.add_argument("--equal_vote_weights_only", action="store_true")
    parser.add_argument("--allow_registry_all_mode", action="store_true", help="Allow per-label registry_all selection during tuning; high recall but can overfit")
    args = parser.parse_args()

    cached_variants = load_cached_variants(args.cache_json)
    variants = parse_variants(args.variant)
    if not cached_variants and not variants:
        variants = {"default": prediction_files(args.pred, args.pred_dir)}
    gold_paths = [Path(path).expanduser().resolve() for path in args.gold]
    if not cached_variants and not gold_paths:
        raise ValueError("Pass --gold with predictions, or pass --cache_json for offline tuning")

    variant_reports = {}
    best_variant_name = ""
    best_variant_score = (-1.0, -1.0, -1.0)
    variant_inputs: dict[str, dict[str, Any]] = {}
    for variant_name, examples in cached_variants.items():
        variant_inputs[variant_name] = {"examples": examples, "warnings": [], "prediction_files": []}
    for variant_name, pred_paths in variants.items():
        examples, warnings = aligned_examples(gold_paths, pred_paths)
        variant_inputs[variant_name] = {"examples": examples, "warnings": warnings, "prediction_files": pred_paths}

    for variant_name, variant_input in variant_inputs.items():
        examples = variant_input["examples"]
        warnings = variant_input["warnings"]
        pred_paths = variant_input["prediction_files"]
        train, holdout = split_examples(examples, args.train_fraction, args.seed)
        if not train:
            raise ValueError(f"Variant {variant_name!r} has no aligned training examples")
        if len(examples) < 30:
            print(
                f"[warning] Variant {variant_name!r} has only {len(examples)} aligned examples. "
                "This is fine for a smoke test, but use 30-50 for calibration and a separate dev-check/test split before trusting the config."
            )
        label_configs = tune_label_configs(
            train,
            prefer=args.prefer,
            equal_vote_weights_only=args.equal_vote_weights_only,
            allow_registry_all_mode=args.allow_registry_all_mode,
        )
        config = build_config(label_configs, variant_name, args)
        train_metrics = evaluate_config(train, config)
        holdout_metrics = evaluate_config(holdout, config) if holdout else None
        all_metrics = evaluate_config(examples, config)
        score = score_tuple(all_metrics["micro"], args.prefer)
        variant_reports[variant_name] = {
            "prediction_files": [str(path) for path in pred_paths],
            "aligned_examples": len(examples),
            "train_examples": len(train),
            "holdout_examples": len(holdout),
            "alignment_warnings": warnings,
            "config": config,
            "train_token_metrics": train_metrics,
            "holdout_token_metrics": holdout_metrics,
            "all_token_metrics": all_metrics,
        }
        if score > best_variant_score:
            best_variant_name = variant_name
            best_variant_score = score

    best_config = dict(variant_reports[best_variant_name]["config"])
    best_config["selected_variant"] = best_variant_name
    best_config["variant_reports"] = variant_reports
    best_config["usage"] = {
        "run_new_data": "--final_selection_policy calibrated --calibration_config /path/to/labelcentered_calibration.json",
        "boundary_normalization": "To tune boundary normalization, run predictions twice with and without --normalize_candidate_boundaries and pass them as --variant base=DIR --variant normalized=DIR.",
    }

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Best variant: {best_variant_name}")
    print(format_micro("All", variant_reports[best_variant_name]["all_token_metrics"]))
    if variant_reports[best_variant_name]["holdout_token_metrics"]:
        print(format_micro("Holdout", variant_reports[best_variant_name]["holdout_token_metrics"]))
    print(f"Calibration config -> {out_path}")


def parse_variants(rows: Sequence[str]) -> dict[str, list[Path]]:
    variants = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"Variant must be NAME=prediction_dir, got: {row}")
        name, directory = row.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Variant name is empty: {row}")
        variants[name] = prediction_files([], [directory])
    return variants


def load_cached_variants(paths: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    variants = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Calibration cache must be a JSON object: {path}")
        variant_name = str(payload.get("variant", path.stem))
        variants[variant_name] = deserialize_examples(payload.get("examples", []))
    return variants


def prediction_files(explicit: Sequence[str], directories: Sequence[str]) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in explicit]
    for directory in directories:
        paths.extend(sorted(Path(directory).expanduser().resolve().glob("*.labelcentered.json")))
    unique = []
    seen = set()
    for path in paths:
        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))
    if not unique:
        raise FileNotFoundError("No prediction JSON files found")
    return unique


def aligned_examples(gold_paths: Sequence[Path], pred_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    predictions = load_prediction_payloads(pred_paths)
    examples = []
    warnings = []
    for gold_path in gold_paths:
        gold_records = load_records(gold_path)
        scoped_predictions = [
            item for item in predictions if str(item[1].get("source_file", "")) == gold_path.name
        ] or predictions
        aligned, align_warnings = align_records(gold_records, scoped_predictions)
        warnings.extend(f"{gold_path.name}: {warning}" for warning in align_warnings)
        for gold_index, record, prediction_path, payload in aligned:
            source_text = build_model_input_from_record(dict(record))
            gold_spans, mismatches, repairs = gold_spans_from_record(record, source_text)
            examples.append({
                "gold_path": str(gold_path),
                "gold_index": gold_index,
                "record_name": record_name(dict(record), gold_index),
                "prediction_path": str(prediction_path),
                "payload": payload,
                "tokens": tokenize_with_offsets(source_text),
                "gold_spans": gold_spans,
                "gold_mismatches": mismatches,
                "gold_repairs": repairs,
            })
    return examples, warnings


def serialize_examples(examples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        payload = example["payload"]
        rows.append({
            "gold_path": example["gold_path"],
            "gold_index": example["gold_index"],
            "record_name": example["record_name"],
            "prediction_path": example["prediction_path"],
            "tokens": [token.__dict__ for token in example["tokens"]],
            "gold_spans": [span.__dict__ for span in example["gold_spans"]],
            "label_candidate_registries": payload.get("label_candidate_registries", {}),
            "candidate_vote_matrices_by_label": payload.get("candidate_vote_matrices_by_label", {}),
            "recommended_candidate_ids_by_label": payload.get("recommended_candidate_ids_by_label", {}),
            "alternative_candidate_ids_by_mode": payload.get("alternative_candidate_ids_by_mode", {}),
        })
    return rows


def deserialize_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        payload = {
            "label_candidate_registries": row.get("label_candidate_registries", {}),
            "candidate_vote_matrices_by_label": row.get("candidate_vote_matrices_by_label", {}),
            "recommended_candidate_ids_by_label": row.get("recommended_candidate_ids_by_label", {}),
            "alternative_candidate_ids_by_mode": row.get("alternative_candidate_ids_by_mode", {}),
        }
        examples.append({
            "gold_path": str(row.get("gold_path", "")),
            "gold_index": int(row.get("gold_index", 0)),
            "record_name": str(row.get("record_name", "")),
            "prediction_path": str(row.get("prediction_path", "")),
            "payload": payload,
            "tokens": [EvalToken(int(token["start"]), int(token["end"]), str(token["text"])) for token in row.get("tokens", [])],
            "gold_spans": [
                EvalSpan(
                    label=str(span["label"]),
                    start=int(span["start"]),
                    end=int(span["end"]),
                    text=str(span["text"]),
                    candidate_id=str(span.get("candidate_id", "")),
                )
                for span in row.get("gold_spans", [])
            ],
            "gold_mismatches": [],
            "gold_repairs": [],
        })
    return examples


def split_examples(examples: Sequence[dict[str, Any]], train_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_fraction >= 1.0:
        return list(examples), []
    if train_fraction <= 0.0:
        raise ValueError("--train_fraction must be greater than 0")
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    split = max(1, int(round(len(rows) * train_fraction)))
    return rows[:split], rows[split:]


def tune_label_configs(
    examples: Sequence[dict[str, Any]],
    *,
    prefer: str,
    equal_vote_weights_only: bool,
    allow_registry_all_mode: bool,
) -> dict[str, dict[str, Any]]:
    output = {}
    for label in LABELS:
        best_config = None
        best_score = (-1.0, -1.0, -1.0)
        for config in config_grid(
            label,
            equal_vote_weights_only=equal_vote_weights_only,
            allow_registry_all_mode=allow_registry_all_mode,
        ):
            metrics = evaluate_label_config(examples, label, config)
            score = score_tuple(metrics, prefer)
            if score > best_score:
                best_config = {**config, "calibration_metrics": metrics}
                best_score = score
        output[label] = best_config or default_label_config(label)
    return output


def config_grid(label: str, *, equal_vote_weights_only: bool, allow_registry_all_mode: bool) -> list[dict[str, Any]]:
    profiles = {"equal": VOTE_WEIGHT_PROFILES["equal"]} if equal_vote_weights_only else VOTE_WEIGHT_PROFILES
    rows = [fixed_mode_config(label, mode) for mode in (
        "recommended",
        "expert_consensus_verified",
        "pubmedbert_plus_verified",
        "registry_filtered",
    )]
    if allow_registry_all_mode:
        rows.append(fixed_mode_config(label, "registry_all"))
    for profile_name, vote_weights in profiles.items():
        for min_vote_fraction in (0.34, 0.50, 0.67, 1.0):
            for min_expert_support in (1, 2, 3):
                for min_confidence in (0.0, 0.25, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
                    for use_rescue in (False, True):
                        rows.append({
                            "label": label,
                            "source_mode": "calibrated_rule",
                            "selector_type": "lightweight_candidate_selector",
                            "require_label_filter": True,
                            "reject_weak_fragments": True,
                            "min_vote_fraction": min_vote_fraction,
                            "min_expert_support": min_expert_support,
                            "min_confidence": min_confidence,
                            "use_high_quality_rescue": use_rescue,
                            "vote_weight_profile": profile_name,
                            "vote_weights": vote_weights,
                        })
    return rows


def fixed_mode_config(label: str, mode: str) -> dict[str, Any]:
    return {
        "label": label,
        "source_mode": mode,
        "selector_type": "fixed_selection_mode",
        "require_label_filter": mode in {"registry_filtered"},
        "reject_weak_fragments": False,
        "min_vote_fraction": 0.0,
        "min_expert_support": 0,
        "min_confidence": 0.0,
        "use_high_quality_rescue": False,
        "vote_weight_profile": "none",
        "vote_weights": VOTE_WEIGHT_PROFILES["equal"],
    }


def default_label_config(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "source_mode": "calibrated_rule",
        "selector_type": "lightweight_candidate_selector",
        "require_label_filter": True,
        "reject_weak_fragments": True,
        "min_vote_fraction": 0.67,
        "min_expert_support": 2,
        "min_confidence": 0.0,
        "use_high_quality_rescue": True,
        "vote_weight_profile": "equal",
        "vote_weights": VOTE_WEIGHT_PROFILES["equal"],
    }


def evaluate_label_config(
    examples: Sequence[dict[str, Any]],
    label: str,
    label_config: Mapping[str, Any],
) -> dict[str, Any]:
    documents = []
    for example in examples:
        selected = select_for_example(example, label, label_config)
        predicted = eval_spans_for_ids(example["payload"], {label: selected})
        documents.append((example["tokens"], example["gold_spans"], predicted))
    metrics = token_level_metrics(documents)[label]
    metrics["selected_candidate_count"] = sum(len(select_for_example(example, label, label_config)) for example in examples)
    return metrics


def select_for_example(example: Mapping[str, Any], label: str, label_config: Mapping[str, Any]) -> list[str]:
    return select_label_candidate_ids(example["payload"], label, label_config)


def evaluate_config(examples: Sequence[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    documents = []
    for example in examples:
        ids_by_label = {}
        for label in LABELS:
            label_config = config["labels"][label]
            ids_by_label[label] = select_for_example(example, label, label_config)
        predicted = eval_spans_for_ids(example["payload"], ids_by_label)
        documents.append((example["tokens"], example["gold_spans"], predicted))
    return token_level_metrics(documents)


def eval_spans_for_ids(payload: Mapping[str, Any], ids_by_label: Mapping[str, Sequence[str]]) -> list[EvalSpan]:
    registry = {
        str(candidate["candidate_id"]): candidate
        for rows in payload.get("label_candidate_registries", {}).values()
        for candidate in rows
    }
    rows = []
    for label in LABELS:
        for candidate_id in ids_by_label.get(label, []):
            candidate = registry.get(candidate_id)
            if candidate:
                rows.append(EvalSpan(
                    label=label,
                    start=int(candidate["start"]),
                    end=int(candidate["end"]),
                    text=str(candidate["text"]),
                    candidate_id=str(candidate_id),
                ))
    return rows


def build_config(label_configs: Mapping[str, Mapping[str, Any]], variant_name: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_name": "calibrated_lightweight_candidate_selector",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_variant": variant_name,
        "tuning_target": args.prefer,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "labels": {label: dict(label_configs[label]) for label in LABELS},
    }


def score_tuple(metrics: Mapping[str, Any], prefer: str) -> tuple[float, float, float]:
    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    if prefer == "precision":
        return precision, f1, recall
    if prefer == "recall":
        return recall, f1, precision
    return f1, recall, precision


def format_micro(prefix: str, metrics: Mapping[str, Any]) -> str:
    micro = metrics["micro"]
    return (
        f"{prefix} token micro: "
        f"P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
