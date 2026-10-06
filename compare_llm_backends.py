"""Compare final Labelcentered spans across configured LLM backends."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Labelcentered.evaluate_labelcentered import evaluate, prediction_paths, readable_report
from Labelcentered.settings import LABELS, LLM_BACKENDS, llm_backend_spec


SUMMARY_FIELDS = [
    "backend",
    "model",
    "prediction_dir",
    "prediction_payload_count",
    "aligned_record_count",
    "overall_token_precision",
    "overall_token_recall",
    "overall_token_f1",
    "overall_token_accuracy",
    "overall_token_one_vs_rest_accuracy",
    "macro_token_precision",
    "macro_token_recall",
    "macro_token_f1",
    "macro_token_accuracy",
]


def backend_prediction_dir(outputs_root: Path, backend: str) -> Path:
    spec = llm_backend_spec(backend)
    return outputs_root / spec.output_dir.name


def summarize_backend(
    backend: str,
    prediction_dir: Path,
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    spec = llm_backend_spec(backend)
    token = report["final_token_level_metrics"]
    micro = token["micro"]
    macro = token["macro"]
    assignment_accuracy = report["final_token_assignment_accuracy"]
    row: Dict[str, Any] = {
        "backend": backend,
        "model": spec.model,
        "prediction_dir": str(prediction_dir),
        "prediction_payload_count": report["prediction_payload_count"],
        "aligned_record_count": report["aligned_record_count"],
        "overall_token_precision": micro["precision"],
        "overall_token_recall": micro["recall"],
        "overall_token_f1": micro["f1"],
        "overall_token_accuracy": assignment_accuracy["accuracy"],
        "overall_token_one_vs_rest_accuracy": micro["accuracy"],
        "macro_token_precision": macro["precision"],
        "macro_token_recall": macro["recall"],
        "macro_token_f1": macro["f1"],
        "macro_token_accuracy": macro["accuracy"],
    }
    for label in LABELS:
        label_metrics = token[label]
        prefix = label.lower()
        row[f"{prefix}_precision"] = label_metrics["precision"]
        row[f"{prefix}_recall"] = label_metrics["recall"]
        row[f"{prefix}_f1"] = label_metrics["f1"]
        row[f"{prefix}_accuracy"] = label_metrics["accuracy"]
    return row


def fieldnames(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    label_fields = [
        f"{label.lower()}_{metric}"
        for label in LABELS
        for metric in ("precision", "recall", "f1", "accuracy")
    ]
    return SUMMARY_FIELDS + label_fields


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})



def comparison_text(rows: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "LLM BACKEND TOKEN-LEVEL COMPARISON",
        "",
        "backend        aligned  precision  recall  f1      accuracy",
    ]
    for row in rows:
        lines.append(
            f"{row['backend']:<14} "
            f"{row['aligned_record_count']:<8} "
            f"{row['overall_token_precision']:.4f}     "
            f"{row['overall_token_recall']:.4f}  "
            f"{row['overall_token_f1']:.4f}  "
            f"{row['overall_token_accuracy']:.4f}"
        )
    if failures:
        lines.extend(["", "FAILURES"])
        for failure in failures:
            lines.append(f"- {failure['backend']}: {failure['error']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare final selected spans from each LLM backend."
    )
    parser.add_argument("--gold", required=True, help="Label Studio gold JSON file")
    parser.add_argument("--outputs_root", default="./outputs")
    parser.add_argument("--backends", nargs="+", choices=sorted(LLM_BACKENDS), default=list(LLM_BACKENDS))
    parser.add_argument("--report_dir", default="./outputs/comparison")
    parser.add_argument("--iou_thresholds", nargs="*", type=float, default=[0.5, 0.75])
    args = parser.parse_args()

    gold_path = Path(args.gold).expanduser().resolve()
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for backend in args.backends:
        prediction_dir = backend_prediction_dir(outputs_root, backend)
        try:
            paths = prediction_paths([], str(prediction_dir))
            if not paths:
                raise FileNotFoundError(f"No .labelcentered.json files in {prediction_dir}")
            report = evaluate(gold_path, paths, args.iou_thresholds)
            backend_json = prediction_dir / "evaluation.json"
            backend_txt = prediction_dir / "evaluation.txt"
            backend_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            backend_txt.write_text(readable_report(report), encoding="utf-8")
            rows.append(summarize_backend(backend, prediction_dir, report))
        except Exception as exc:
            failures.append({"backend": backend, "error": str(exc)})

    summary = {
        "gold_path": str(gold_path),
        "outputs_root": str(outputs_root),
        "backends": list(args.backends),
        "rows": rows,
        "failures": failures,
    }
    summary_json = report_dir / "llm_backend_comparison.json"
    summary_csv = report_dir / "llm_backend_comparison.csv"
    summary_txt = report_dir / "llm_backend_comparison.txt"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(summary_csv, rows)
    text = comparison_text(rows, failures)
    summary_txt.write_text(text, encoding="utf-8")
    print(text)
    print(f"JSON summary -> {summary_json}")
    print(f"CSV summary -> {summary_csv}")
    print(f"Text summary -> {summary_txt}")
    if failures:
        raise SystemExit(1)
    if not rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
