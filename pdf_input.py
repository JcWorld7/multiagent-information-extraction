"""PDF text extraction and Methods/Results model-input construction."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

METHOD_PATTERN = (
    r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?"
    r"(methods?|methodology|experimental\s+design)\s*\n"
    r"(.*?)(?=\n\s*(?:\d+\.?\s*)?(results?|discussion|analysis|conclusion))"
)
RESULTS_PATTERN = (
    r"(?i)(?:^|\n)\s*(?:\d+\.?\s*)?"
    r"(results?|findings|outcomes)\s*\n"
    r"(.*?)(?=\n\s*(?:\d+\.?\s*)?(discussion|conclusion|analysis|limitations))"
)


def extract_full_text_pymupdf(pdf_path: Path) -> str:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()


def extract_methods_results(pdf_path: Path) -> Dict[str, str]:
    full_text = extract_full_text_pymupdf(pdf_path)
    method_text = ""
    results_text = ""
    method_match = re.search(METHOD_PATTERN, full_text, re.DOTALL)
    if method_match:
        method_text = method_match.group(2).strip()
    results_match = re.search(RESULTS_PATTERN, full_text, re.DOTALL)
    if results_match:
        results_text = results_match.group(2).strip()
    return {"method": method_text, "results": results_text, "full_text": full_text}


def build_model_input_from_pdf(pdf_path: Path) -> str:
    sections = extract_methods_results(pdf_path)
    method = sections.get("method", "").strip()
    results = sections.get("results", "").strip()
    method_section = f"=== METHODS ===\n{method}" if method else ""
    results_section = f"=== RESULTS ===\n{results}" if results else ""
    return method_section + "\n\n" + results_section

