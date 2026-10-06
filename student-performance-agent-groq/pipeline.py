from __future__ import annotations

from data.calculations import calculate_metrics
from data.validation import read_upload, validate_and_clean


def prepare_upload(filename: str, content: bytes, pass_mark: float = 50.0):
    """Read, validate, normalize, and calculate an uploaded marks file."""
    raw = read_upload(filename, content)
    clean, errors, warnings = validate_and_clean(raw)
    if errors:
        return {"metrics": None, "class_summary": None, "errors": errors, "warnings": warnings}
    metrics, class_summary = calculate_metrics(clean, pass_mark)
    return {"metrics": metrics, "class_summary": class_summary, "errors": errors, "warnings": warnings}
