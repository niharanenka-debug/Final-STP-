# Groq 8,000 TPM implementation

This Groq edition is designed for a maximum budget of **8,000 tokens per minute**.

## Whole-class execution

The class pipeline never sends the full dataset in one request:

```text
records
  → compact records
  → token estimate
  → sequential 4,000-token chunks
  → one compact JSON result per chunk
  → compact aggregation request
  → class analysis
```

Each chunk contains only:

- student ID
- subject
- percentage
- grade
- status
- change from the previous assessment

Raw names, maximum marks, and repeated upload metadata are not sent to class analysis.

## Rate protection

`llm_gateway.py`:

- estimates input tokens with `tiktoken`, with a character-based fallback
- reserves estimated input plus output tokens before every call
- tracks a rolling 60-second token window
- waits when the next request would exceed 8,000 TPM
- sends requests sequentially
- caps ordinary requests below 6,000 estimated total tokens
- handles 429 and 413 responses without replaying the same request
- waits for the window, reduces input by 45%, reduces output, and retries

## Configuration

```env
GROQ_TPM_LIMIT=8000
GROQ_CHUNK_INPUT_TOKENS=4000
GROQ_MAX_REQUEST_TOKENS=6000
GROQ_MAX_OUTPUT_TOKENS=700
GROQ_MAX_RETRIES=4
```

Individual reports use one combined analysis/study-plan request plus one report request by default. The quality-check request is optional and disabled by default.
