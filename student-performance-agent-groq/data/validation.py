from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd


ALIASES = {
    "student_id": ["student_id", "studentid", "id", "roll_no", "roll_number", "admission_no"],
    "student_name": ["student_name", "studentname", "name", "full_name", "student"],
    "class_name": ["class_name", "class", "section", "grade_level"],
    "subject": ["subject", "course", "paper", "module"],
    "exam": ["exam", "assessment", "test", "exam_name"],
    "term": ["term", "semester", "period"],
    "marks_obtained": ["marks_obtained", "marks", "score", "obtained", "marks_scored"],
    "max_marks": ["max_marks", "maximum_marks", "total_marks", "max_score", "out_of", "total"],
    "attendance_percent": ["attendance_percent", "attendance", "attendance_percentage"],
}
REQUIRED = ["student_id", "subject", "marks_obtained", "max_marks"]


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def read_upload(filename: str, content: bytes) -> pd.DataFrame:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "csv":
        return pd.read_csv(io.BytesIO(content))
    if suffix in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(content))
    raise ValueError("Unsupported file type. Upload a CSV or Excel (.xlsx/.xls) file.")


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rename: dict[str, str] = {}
    reverse = {_key(col): col for col in df.columns}
    warnings: list[str] = []
    for canonical, options in ALIASES.items():
        for option in options:
            if option in reverse:
                rename[reverse[option]] = canonical
                break
    result = df.rename(columns=rename).copy()
    result.columns = [_key(c) for c in result.columns]
    missing = [c for c in REQUIRED if c not in result.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Required fields are student_id, subject, marks_obtained, and max_marks."
        )
    optional_defaults = {"student_name": "", "class_name": "", "exam": "", "term": "", "attendance_percent": None}
    for column, default in optional_defaults.items():
        if column not in result.columns:
            result[column] = default
            warnings.append(f"Optional column '{column}' was not provided; defaults were used.")
    return result, warnings


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    result, warnings = normalize_columns(df)
    errors: list[str] = []
    for col in ["student_id", "student_name", "class_name", "subject", "exam", "term"]:
        result[col] = result[col].fillna("").astype(str).str.strip()
    for col in ["marks_obtained", "max_marks", "attendance_percent"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if result["student_id"].eq("").any():
        errors.append(f"{int(result['student_id'].eq('').sum())} row(s) have a blank student_id.")
    if result["subject"].eq("").any():
        errors.append(f"{int(result['subject'].eq('').sum())} row(s) have a blank subject.")
    if result["marks_obtained"].isna().any():
        errors.append(f"{int(result['marks_obtained'].isna().sum())} row(s) have non-numeric or missing marks.")
    if result["max_marks"].isna().any():
        errors.append(f"{int(result['max_marks'].isna().sum())} row(s) have non-numeric or missing max_marks.")
    valid_numeric = result["marks_obtained"].notna() & result["max_marks"].notna()
    if (result.loc[valid_numeric, "max_marks"] <= 0).any():
        errors.append("max_marks must be greater than zero.")
    if (result.loc[valid_numeric, "marks_obtained"] < 0).any():
        errors.append("marks_obtained cannot be negative.")
    if (result.loc[valid_numeric, "marks_obtained"] > result.loc[valid_numeric, "max_marks"]).any():
        errors.append("Some rows have marks_obtained greater than max_marks.")
    if result["attendance_percent"].notna().any() and ((result["attendance_percent"] < 0) | (result["attendance_percent"] > 100)).any():
        errors.append("attendance_percent must be between 0 and 100.")
    duplicate_count = int(result.duplicated(subset=["student_id", "subject", "exam", "term"], keep=False).sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} row(s) may be duplicates for the same student, subject, exam, and term.")
    result = result.dropna(subset=["student_id", "subject", "marks_obtained", "max_marks"]).copy()
    result["marks_obtained"] = result["marks_obtained"].round(2)
    result["max_marks"] = result["max_marks"].round(2)
    result["percentage"] = (result["marks_obtained"] / result["max_marks"] * 100).round(2)
    return result, errors, warnings
