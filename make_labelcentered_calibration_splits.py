"""Create fast calibration/dev/test JSON splits from annotated Label Studio data."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.json_input import load_records
from Labelcentered.settings import DEFAULT_DATASET_PATHS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dev-small/dev-check/test JSON files for fast calibration."
    )
    parser.add_argument("--json", nargs="*", help="Annotated JSON files; defaults to settings.DEFAULT_DATASET_PATHS")
    parser.add_argument("--out_dir", default="./labelcentered_calibration_splits")
    parser.add_argument("--dev_small", type=int, default=50, help="Number of records for fast calibration")
    parser.add_argument("--dev_check", type=int, default=50, help="Number of records for stability check")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    input_paths = [Path(path).expanduser().resolve() for path in args.json] if args.json else list(DEFAULT_DATASET_PATHS)
    records, missing = load_records_with_source(input_paths)
    if not records:
        raise FileNotFoundError("No records found. On IDAS, confirm DEFAULT_DATASET_PATHS exist or pass --json explicitly.")
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)

    dev_small = records[: args.dev_small]
    dev_check = records[args.dev_small : args.dev_small + args.dev_check]
    test = records[args.dev_small + args.dev_check :]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "dev_small": write_split(out_dir / "dev_small.json", dev_small),
        "dev_check": write_split(out_dir / "dev_check.json", dev_check),
        "test": write_split(out_dir / "test.json", test),
    }
    manifest = {
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "input_files": [str(path) for path in input_paths],
        "missing_input_files": [str(path) for path in missing],
        "total_records": len(records),
        "splits": {
            "dev_small": {"records": len(dev_small), "path": str(split_paths["dev_small"])},
            "dev_check": {"records": len(dev_check), "path": str(split_paths["dev_check"])},
            "test": {"records": len(test), "path": str(split_paths["test"])},
        },
        "recommended_workflow": [
            "Run run_json_labelcentered.py once on dev_small.json and optionally dev_check.json/test.json.",
            "Build offline caches with build_labelcentered_calibration_cache.py.",
            "Tune repeatedly from --cache_json without rerunning BERT or LLM agents.",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Records: {len(records)}")
    print(f"dev_small: {len(dev_small)} -> {split_paths['dev_small']}")
    print(f"dev_check: {len(dev_check)} -> {split_paths['dev_check']}")
    print(f"test: {len(test)} -> {split_paths['test']}")
    if missing:
        print(f"Missing input files skipped: {len(missing)}")
    print(f"Manifest -> {manifest_path}")


def load_records_with_source(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    records = []
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(path)
            continue
        for index, record in enumerate(load_records(path)):
            row = dict(record)
            row.setdefault("labelcentered_split_source", {"source_file": str(path), "source_index": index})
            records.append(row)
    return records, missing


def write_split(path: Path, records: Sequence[MappingLike]) -> Path:
    path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


MappingLike = dict[str, Any]


if __name__ == "__main__":
    main()

