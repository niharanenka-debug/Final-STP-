from __future__ import annotations

import pandas as pd


def grade_for(percentage: float) -> str:
    if percentage >= 90: return "A+"
    if percentage >= 80: return "A"
    if percentage >= 70: return "B"
    if percentage >= 60: return "C"
    if percentage >= 50: return "D"
    return "F"


def calculate_metrics(df: pd.DataFrame, pass_mark: float = 50.0) -> tuple[pd.DataFrame, dict]:
    result = df.copy()
    result["grade"] = result["percentage"].map(grade_for)
    result["status"] = result["percentage"].apply(lambda x: "Pass" if x >= pass_mark else "Needs support")
    subject_avg = result.groupby("subject", as_index=False)["percentage"].mean().rename(columns={"percentage": "class_subject_average"})
    result = result.merge(subject_avg, on="subject", how="left")
    result["difference_from_subject_average"] = (result["percentage"] - result["class_subject_average"]).round(2)
    if "term" in result.columns and result["term"].nunique() > 1:
        result = result.sort_values(["student_id", "subject", "term"])
        result["previous_percentage"] = result.groupby(["student_id", "subject"])["percentage"].shift(1)
        result["change_from_previous"] = (result["percentage"] - result["previous_percentage"]).round(2)
    else:
        result["previous_percentage"] = None
        result["change_from_previous"] = None
    class_summary = {
        "student_count": int(result["student_id"].nunique()),
        "record_count": int(len(result)),
        "overall_average": round(float(result["percentage"].mean()), 2) if len(result) else 0,
        "pass_rate": round(float((result["percentage"] >= pass_mark).mean() * 100), 2) if len(result) else 0,
        "subject_averages": {
            str(row.subject): round(float(row.class_subject_average), 2)
            for row in subject_avg.itertuples()
        },
        "grade_distribution": {str(k): int(v) for k, v in result["grade"].value_counts().to_dict().items()},
    }
    return result, class_summary


def student_snapshot(metrics: pd.DataFrame, student_id: str, pass_mark: float = 50.0) -> dict:
    rows = metrics[metrics["student_id"].astype(str) == str(student_id)].copy()
    if rows.empty:
        raise ValueError(f"No records found for student {student_id}.")
    latest_term = rows["term"].iloc[-1] if "term" in rows.columns else ""
    latest = rows[rows["term"] == latest_term] if latest_term else rows
    return {
        "student_id": str(student_id),
        "student_name": str(rows["student_name"].iloc[0]),
        "class_name": str(rows["class_name"].iloc[0]),
        "latest_term": str(latest_term),
        "overall_average": round(float(latest["percentage"].mean()), 2),
        "pass_rate": round(float((latest["percentage"] >= pass_mark).mean() * 100), 2),
        "subjects": latest[["subject", "marks_obtained", "max_marks", "percentage", "grade", "status", "class_subject_average", "difference_from_subject_average", "change_from_previous"]].to_dict("records"),
        "history": rows[["subject", "term", "percentage", "grade", "change_from_previous"]].to_dict("records"),
    }
