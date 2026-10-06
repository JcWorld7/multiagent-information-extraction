"""Apply a calibrated Labelcentered selection config to prediction JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.calibration import (
    candidate_ids_by_label_from_config,
    load_calibration_config,
    spans_for_candidate_ids,
)
from Labelcentered.global_meta_agent import detect_cross_label_conflicts
from Labelcentered.json_input import safe_name
from Labelcentered.output_validation import validate_output
from Labelcentered.settings import LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply calibrated selection to Labelcentered outputs.")
    parser.add_argument("--config", required=True, help="Calibration JSON from tune_labelcentered_calibration.py")
    parser.add_argument("--pred", nargs="*", default=[], help="Input .labelcentered.json files")
    parser.add_argument("--pred_dir", nargs="*", default=[], help="Input directories")
    parser.add_argument("--out_dir", required=True, help="Directory for calibrated copies")
    parser.add_argument("--suffix", default="calibrated")
    args = parser.parse_args()

    config = load_calibration_config(args.config)
    paths = prediction_files(args.pred, args.pred_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = {"processed": 0, "failed": 0}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibrated = apply_config(payload, config)
            errors = validate_output(calibrated)
            if errors:
                raise ValueError(f"Output validation failed: {errors}")
            out_path = out_dir / calibrated_filename(path, args.suffix)
            out_path.write_text(json.dumps(calibrated, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved -> {out_path}")
            totals["processed"] += 1
        except Exception as exc:
            print(f"[error] {path}: {exc}")
            totals["failed"] += 1
    print(totals)
    if totals["failed"]:
        raise SystemExit(1)


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


def apply_config(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    final_ids_by_label = candidate_ids_by_label_from_config(output, config)
    final_spans_by_label = spans_for_candidate_ids(output, final_ids_by_label)
    final_spans = [span for label in LABELS for span in final_spans_by_label[label]]
    output["final_selection_policy"] = "calibrated"
    output["calibration_config_summary"] = {
        "schema_version": config.get("schema_version"),
        "policy_name": config.get("policy_name"),
        "selected_variant": config.get("selected_variant"),
        "tuning_target": config.get("tuning_target"),
    }
    output["final_candidate_ids_by_label"] = final_ids_by_label
    output["final_spans_by_label"] = final_spans_by_label
    output["final_spans"] = final_spans
    output["missing_labels"] = [label for label in LABELS if not final_ids_by_label.get(label)]
    output["cross_label_conflicts"] = detect_cross_label_conflicts(final_spans_by_label)
    output["overall_consensus_reached"] = False
    output["global_meta_decision"] = {
        "final_candidate_ids_by_label": final_ids_by_label,
        "final_spans_by_label": final_spans_by_label,
        "final_spans": final_spans,
        "cross_label_conflicts": output["cross_label_conflicts"],
        "rationale": "assembled calibrated candidate IDs without rewriting spans",
    }
    return output


def calibrated_filename(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(".labelcentered.json"):
        return name[: -len(".labelcentered.json")] + f".{safe_name(suffix)}.labelcentered.json"
    return path.stem + f".{safe_name(suffix)}.json"


if __name__ == "__main__":
    main()

