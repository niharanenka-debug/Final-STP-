from __future__ import annotations

from typing import Any

from llm_gateway import GroqClient, chunk_records, compact_json, estimate_tokens

client = GroqClient()

ANALYSIS_SYSTEM = (
    "You are a school performance analyst. Use only supplied evidence; do not diagnose. "
    "Return compact JSON with strengths, weaknesses, trends, data_notes, confidence. "
    "Maximum 3 items per array; each item must include evidence."
)
RECOMMENDATION_SYSTEM = (
    "You are a personalized study planner. Use only supplied evidence. Do not diagnose or invent facts. "
    "Return compact JSON with priority_subjects, weekly_plan, study_strategies, measurable_targets, "
    "review_schedule, motivation_note, limitations. Maximum 3 priority subjects and 5 weekly activities."
)


def analyze_student(snapshot: dict, class_summary: dict) -> dict:
    return client.json_complete(ANALYSIS_SYSTEM, f"STUDENT:{compact_json(snapshot)}\nCLASS:{compact_json(class_summary)}", max_output_tokens=450)


def generate_personalized_recommendations(snapshot: dict, analysis: dict) -> dict:
    return client.json_complete(RECOMMENDATION_SYSTEM, f"STUDENT:{compact_json(snapshot)}\nANALYSIS:{compact_json(analysis)}", max_output_tokens=550)


def analyze_and_recommend_student(snapshot: dict, class_summary: dict) -> dict:
    """Low-cost individual path: analysis and study plan in one request."""
    system = (
        "You are a school performance and study-planning assistant. Use only evidence; do not diagnose. "
        "Return exactly {analysis,recommendations}. analysis has strengths, weaknesses, trends, data_notes, confidence. "
        "recommendations has priority_subjects, weekly_plan, study_strategies, measurable_targets, review_schedule, "
        "motivation_note, limitations. Maximum 3 items per array and 5 weekly activities. Compact JSON only."
    )
    return client.json_complete(system, f"STUDENT:{compact_json(snapshot)}\nCLASS:{compact_json(class_summary)}", max_output_tokens=850)


def _compact_record(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("student_id", "subject", "percentage", "grade", "status", "change_from_previous") if row.get(key) is not None}


def _analyze_chunk(chunk: list[dict[str, Any]], chunk_number: int, total_chunks: int) -> dict[str, Any]:
    system = (
        "Analyze one class-data chunk. Use only records in this chunk. Return compact JSON with: "
        "chunk_number, subject_findings (max 5 objects), students_needing_support (max 10 IDs with reason), "
        "notable_trends (max 5), data_notes (max 3). Do not calculate unsupported facts or write a report."
    )
    return client.json_complete(system, f"CHUNK {chunk_number}/{total_chunks}\nRECORDS:{compact_json(chunk)}", max_output_tokens=450)


def _aggregate_results(chunk_results: list[dict[str, Any]], class_summary: dict[str, Any]) -> dict[str, Any]:
    system = (
        "Aggregate compact class-analysis results. Use only supplied summaries and class metrics. Return compact JSON "
        "with strengths, weaknesses, priority_students, trends, recommendations, confidence. Maximum 5 items per array. "
        "Do not repeat raw records and do not invent student facts."
    )
    user = f"CLASS_SUMMARY:{compact_json(class_summary)}\nCHUNK_RESULTS:{compact_json(chunk_results)}"
    if estimate_tokens(system) + estimate_tokens(user) + 900 > 6000:
        reduced = [{"students_needing_support": r.get("students_needing_support", [])[:5], "subject_findings": r.get("subject_findings", [])[:3], "notable_trends": r.get("notable_trends", [])[:3]} for r in chunk_results]
        user = f"CLASS_SUMMARY:{compact_json(class_summary)}\nCOMPACT_RESULTS:{compact_json(reduced)}"
    return client.json_complete(system, user, max_output_tokens=700)


def analyze_class(metrics_records: list[dict], class_summary: dict) -> dict:
    compact_records = [_compact_record(row) for row in metrics_records]
    system = "Analyze one class-data chunk. Return compact JSON. Keep subject_findings <=5, students_needing_support <=10, notable_trends <=5, data_notes <=3."
    chunks = chunk_records(compact_records, system, target_tokens=4000)
    results = []
    for number, chunk in enumerate(chunks, start=1):
        try:
            results.append(_analyze_chunk(chunk, number, len(chunks)))
        except Exception as exc:
            # Preserve progress and allow remaining chunks to be analyzed.
            results.append({"chunk_number": number, "status": "failed", "data_notes": [f"Chunk skipped: {type(exc).__name__}"]})
    if not results:
        return {"strengths": [], "weaknesses": [], "priority_students": [], "trends": [], "recommendations": [], "confidence": "low"}
    try:
        return _aggregate_results(results, class_summary)
    except Exception as exc:
        return {"status": "partial", "confidence": "limited", "aggregation_error": type(exc).__name__, "chunk_results": results}


def make_report(scope: str, evidence: dict, analysis: dict, recommendations: dict | None = None) -> str:
    system = "Write a concise Markdown school report from supplied evidence only. Include facts, interpretation, recommendations, and limitations. Do not fabricate or diagnose."
    user = f"SCOPE:{scope}\nEVIDENCE:{compact_json(evidence)}\nANALYSIS:{compact_json(analysis)}\nRECOMMENDATIONS:{compact_json(recommendations or {})}"
    return client.complete(system, user, max_output_tokens=850)


def quality_check(report: str, evidence: dict) -> dict:
    system = "Check report against evidence. Return compact JSON: passed, score, unsupported_claims, contradictions, missing_items, revision_notes. Keep arrays <=3."
    return client.json_complete(system, f"EVIDENCE:{compact_json(evidence)}\nREPORT:{report}", max_output_tokens=350)


def answer_question(question: str, evidence: dict) -> str:
    return client.complete("Answer using only supplied evidence. Be concise and say when data is insufficient.", f"QUESTION:{question}\nEVIDENCE:{compact_json(evidence)}", max_output_tokens=350)
