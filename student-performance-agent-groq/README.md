# Student Performance AI — Groq Edition

This copy is configured for **Groq**. Put your API key in `.env`, then run `streamlit run app.py`.

# Student Performance Multi-Agent System

A Streamlit application that accepts CSV or Excel marks, calculates reliable performance metrics, uses the Groq API for educational interpretation, generates reports, stores results in SQLite, and answers questions about individual students or a whole class.

## Features

- CSV and XLSX upload
- Automatic column normalization and validation
- Deterministic percentage, grade, pass/fail, class-average, and trend calculations
- Individual student analysis
- Whole-class analysis
- Personalized weekly study plans for each student
- Groq-powered strengths, weaknesses, recommendations, and explanations
- Quality-check agent that verifies report claims against calculated data
- SQLite persistence for uploads, rows, metrics, analyses, and reports
- Natural-language questions over the selected dataset
- Downloadable Markdown reports

## Setup

```bash
cd student-performance-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY
streamlit run app.py
```

The app uses `openai/gpt-oss-20b` by default. You can change `GROQ_MODEL` in `.env`.

## Input format

Preferred long format:

```text
student_id,student_name,subject,exam,term,marks_obtained,max_marks
S001,Asha,Mathematics,Midterm,Term 1,82,100
```

Required columns are student identifier, subject, marks obtained, and maximum marks. Names, exam, term, class, and attendance are optional. The validator accepts common alternatives such as `Student ID`, `Name`, `Subject`, `Marks`, `Total`, and `Max Marks`.

## Design

```text
Upload → Validate → Calculate → Performance Analysis → Personalized Study Plan → Report → Quality Check → SQLite
```

Python owns parsing and calculations. Groq is used for explanation, recommendations, report writing, and optional report verification.

## Personalized study recommendations

For an individual student, the Recommendation Agent receives the student's subject scores, class comparisons, trends, attendance when available, and the Performance Agent's findings. It returns:

- Priority subjects with a reason and target score
- A practical weekly plan with day, activity, subject, and duration
- Subject-specific study strategies
- Measurable improvement targets
- A review schedule
- A supportive motivation note
- Limitations where the available data is insufficient

The study plan is shown in the **Recommendation Agent** output panel and is also supplied to the Report Agent. It is intended as educator decision support, not as a diagnosis or a replacement for teacher judgment.

## Privacy

Use student IDs instead of real names when possible. Do not commit `.env`, uploaded files, or the SQLite database to source control. The application is decision support for educators and should not be used as the sole basis for high-impact academic decisions.
