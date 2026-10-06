"""Build a compact offline calibration cache from Labelcentered outputs."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.tune_labelcentered_calibration import (
    aligned_examples,
    prediction_files,
    serialize_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact cache for offline calibration tuning."
    )
    parser.add_argument("--gold", nargs="+", required=True, help="Gold Label Studio JSON files")
    parser.add_argument("--pred", nargs="*", default=[], help="Prediction .labelcentered.json files")
    parser.add_argument("--pred_dir", nargs="*", default=[], help="Prediction directories")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--variant", default="default")
    args = parser.parse_args()

    gold_paths = [Path(path).expanduser().resolve() for path in args.gold]
    pred_paths = prediction_files(args.pred, args.pred_dir)
    examples, warnings = aligned_examples(gold_paths, pred_paths)
    if not examples:
        raise ValueError("No aligned examples found for cache")
    payload = {
        "schema_version": 1,
        "cache_type": "labelcentered_offline_calibration_cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "gold_files": [str(path) for path in gold_paths],
        "prediction_files": [str(path) for path in pred_paths],
        "aligned_examples": len(examples),
        "alignment_warnings": warnings,
        "examples": serialize_examples(examples),
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Aligned examples: {len(examples)}")
    if warnings:
        print(f"Alignment warnings: {len(warnings)}")
    print(f"Calibration cache -> {out_path}")


if __name__ == "__main__":
    main()

