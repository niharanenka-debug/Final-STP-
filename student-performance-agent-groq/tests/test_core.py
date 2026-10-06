import pandas as pd

from data.calculations import calculate_metrics, grade_for
from data.validation import validate_and_clean


def test_grade_boundaries():
    assert grade_for(95) == "A+"
    assert grade_for(80) == "A"
    assert grade_for(50) == "D"
    assert grade_for(49.9) == "F"


def test_validation_and_metrics():
    raw = pd.DataFrame({
        "Student ID": ["S1", "S1"],
        "Name": ["Test", "Test"],
        "Subject": ["Math", "Physics"],
        "Marks": [80, 40],
        "Total": [100, 100],
    })
    clean, errors, _ = validate_and_clean(raw)
    assert errors == []
    metrics, summary = calculate_metrics(clean)
    assert list(metrics["percentage"]) == [80.0, 40.0]
    assert summary["student_count"] == 1
    assert summary["pass_rate"] == 50.0
