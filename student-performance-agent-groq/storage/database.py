from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd


DB_PATH = os.getenv("DATABASE_PATH", "student_performance.db")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS uploads (
            upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            created_at TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            validation_warnings TEXT
        );
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            student_id TEXT, student_name TEXT, class_name TEXT, subject TEXT,
            exam TEXT, term TEXT, marks_obtained REAL, max_marks REAL,
            attendance_percent REAL, percentage REAL, grade TEXT, status TEXT,
            class_subject_average REAL, difference_from_subject_average REAL,
            previous_percentage REAL, change_from_previous REAL,
            FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
        );
        CREATE TABLE IF NOT EXISTS reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER NOT NULL,
            scope TEXT NOT NULL,
            subject_id TEXT,
            report_markdown TEXT NOT NULL,
            analysis_json TEXT,
            quality_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(upload_id) REFERENCES uploads(upload_id)
        );
        """)


def save_upload(filename: str, metrics: pd.DataFrame, warnings: list[str]) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as conn:
        cur = conn.execute("INSERT INTO uploads(filename, created_at, row_count, validation_warnings) VALUES (?, ?, ?, ?)", (filename, now, len(metrics), json.dumps(warnings)))
        upload_id = int(cur.lastrowid)
        cols = ["student_id", "student_name", "class_name", "subject", "exam", "term", "marks_obtained", "max_marks", "attendance_percent", "percentage", "grade", "status", "class_subject_average", "difference_from_subject_average", "previous_percentage", "change_from_previous"]
        rows = []
        for record in metrics[cols].where(pd.notna(metrics[cols]), None).to_dict("records"):
            rows.append((upload_id, *[record.get(c) for c in cols]))
        conn.executemany(f"INSERT INTO records(upload_id, {', '.join(cols)}) VALUES ({', '.join(['?'] * (len(cols)+1))})", rows)
    return upload_id


def save_report(upload_id: int, scope: str, subject_id: str | None, report: str, analysis: dict, quality: dict) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO reports(upload_id, scope, subject_id, report_markdown, analysis_json, quality_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (upload_id, scope, subject_id, report, json.dumps(analysis), json.dumps(quality), datetime.now(timezone.utc).isoformat()))


def recent_reports(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM reports ORDER BY report_id DESC LIMIT ?", (limit,)).fetchall()]
