from llm_gateway import chunk_records, estimate_tokens


def test_chunk_records_respects_target_estimate():
    records = [{"student_id": f"S{i:03d}", "subject": "Math", "percentage": 70 + i % 10} for i in range(200)]
    system = "Analyze one compact class-data chunk and return JSON."
    chunks = chunk_records(records, system, target_tokens=400)
    assert len(chunks) > 1
    assert sum(len(chunk) for chunk in chunks) == len(records)
    for chunk in chunks:
        assert estimate_tokens(system) + estimate_tokens(chunk) <= 500
