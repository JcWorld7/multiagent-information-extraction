"""Label Studio JSON loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def build_model_input_from_record(record: Dict[str, Any]) -> str:
    data = record.get("data", record)
    return (data.get("method_section", "") or "") + "\n\n" + (data.get("results_section", "") or "")


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = None
        for key in ("records", "tasks", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if records is None:
            records = [payload]
    else:
        raise ValueError(f"Top-level JSON must be a list or object, not {type(payload).__name__}")
    invalid = [i for i, item in enumerate(records) if not isinstance(item, dict)]
    if invalid:
        raise ValueError(f"Records at indices {invalid[:10]} are not JSON objects")
    return records


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value)
    return cleaned[:180] or "record"


def record_name(record: Dict[str, Any], index: int) -> str:
    data = record.get("data", record)
    for key in ("file", "filename", "id", "doc_id", "document_id"):
        value = data.get(key)
        if value not in (None, ""):
            return safe_name(str(value))
    return f"rec{index:04d}"


def deduplicate_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    output = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if str(resolved) not in seen:
            seen.add(str(resolved))
            output.append(resolved)
    return output

