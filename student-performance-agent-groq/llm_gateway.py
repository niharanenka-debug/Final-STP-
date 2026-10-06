from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable
import requests
from dotenv import load_dotenv
# ============================================================
# ENVIRONMENT
# ============================================================
load_dotenv()
def config_value(
    name: str,
    default: str = "",
) -> str:
    """
    Read configuration from:
    1. Environment variables
    2. Streamlit secrets
    3. Default value
    """
    value = os.getenv(name)
    if value:
        return str(value).strip()
    try:
        import streamlit as st
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return default
# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_URL = (
    "https://api.groq.com/openai/v1/chat/completions"
)
DEFAULT_OUTPUT_TOKENS = 700
MAX_REQUEST_TOKENS = 6000
MAX_RETRIES = 3
WINDOW_SECONDS = 60
TPM_LIMIT = 8000
# ============================================================
# EXCEPTIONS
# ============================================================
class RequestTooLarge(RuntimeError):
    pass
class RateLimitExceeded(RuntimeError):
    pass
# ============================================================
# TOKEN ESTIMATION
# ============================================================
def estimate_tokens(
    text: str,
) -> int:
    """
    Lightweight token estimator.
    We intentionally avoid depending on tiktoken because
    Groq model tokenizers may differ.
    """
    if not text:
        return 0
    # Rough approximation:
    # 1 token ~= 4 characters for normal English text.
    return max(
        1,
        (len(text) + 3) // 4,
    )
# ============================================================
# JSON COMPRESSION
# ============================================================
def compact_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
# ============================================================
# SIMPLE RATE LIMITER
# ============================================================
@dataclass
class RequestRecord:
    timestamp: float
    tokens: int
class SimpleRateLimiter:
    """
    Simple sequential token limiter.
    This intentionally avoids threading.Lock and the old
    _prune() implementation that caused the previous error.
    """
    def __init__(
        self,
        limit: int = TPM_LIMIT,
        window: int = WINDOW_SECONDS,
    ):
        self.limit = max(
            1,
            int(limit),
        )
        self.window = max(
            1,
            int(window),
        )
        self.events: list[
            RequestRecord
        ] = []
    def cleanup(self) -> None:
        now = time.time()
        cutoff = (
            now - self.window
        )
        self.events = [
            event
            for event in self.events
            if event.timestamp >= cutoff
        ]
    def current_usage(self) -> int:
        self.cleanup()
        return sum(
            event.tokens
            for event in self.events
        )
    def wait_for_capacity(
        self,
        required_tokens: int,
    ) -> None:
        required_tokens = max(
            1,
            int(required_tokens),
        )
        # If a single request is larger than the entire
        # configured budget, it cannot fit.
        if required_tokens > self.limit:
            raise RateLimitExceeded(
                "Requested token budget "
                f"{required_tokens} exceeds "
                f"configured TPM limit "
                f"{self.limit}."
            )
        while True:
            self.cleanup()
            used = self.current_usage()
            if (
                used + required_tokens
                <= self.limit
            ):
                return
            if not self.events:
                return
            oldest = min(
                self.events,
                key=lambda item: item.timestamp,
            )
            wait_seconds = (
                oldest.timestamp
                + self.window
                - time.time()
            )
            if wait_seconds <= 0:
                self.cleanup()
                continue
            time.sleep(
                min(
                    wait_seconds,
                    5.0,
                )
            )
    def reserve(
        self,
        tokens: int,
    ) -> None:
        self.cleanup()
        self.events.append(
            RequestRecord(
                timestamp=time.time(),
                tokens=max(
                    1,
                    int(tokens),
                ),
            )
        )
# ============================================================
# GROQ CLIENT
# ============================================================
class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        url: str | None = None,
        max_output_tokens: int | None = None,
        max_request_tokens: int | None = None,
        max_retries: int | None = None,
    ):
        self.api_key = (
            api_key
            or config_value(
                "GROQ_API_KEY"
            )
            or config_value(
                "LLM_API_KEY"
            )
        )
        self.model = (
            model
            or config_value(
                "GROQ_MODEL",
                DEFAULT_MODEL,
            )
        )
        self.url = (
            url
            or config_value(
                "GROQ_API_URL",
                DEFAULT_URL,
            )
        )
        self.max_output_tokens = int(
            max_output_tokens
            if max_output_tokens is not None
            else config_value(
                "GROQ_MAX_OUTPUT_TOKENS",
                str(DEFAULT_OUTPUT_TOKENS),
            )
        )
        self.max_request_tokens = int(
            max_request_tokens
            if max_request_tokens is not None
            else config_value(
                "GROQ_MAX_REQUEST_TOKENS",
                str(MAX_REQUEST_TOKENS),
            )
        )
        self.max_retries = int(
            max_retries
            if max_retries is not None
            else config_value(
                "GROQ_MAX_RETRIES",
                str(MAX_RETRIES),
            )
        )
        self.limiter = SimpleRateLimiter(
            limit=int(
                config_value(
                    "GROQ_TPM_LIMIT",
                    str(TPM_LIMIT),
                )
            ),
            window=WINDOW_SECONDS,
        )
        self.session = requests.Session()
    # ========================================================
    # API KEY
    # ========================================================
    def _check_api_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "Groq API key is missing.\n\n"
                "Set GROQ_API_KEY in:\n"
                "• Streamlit Cloud Secrets, or\n"
                "• .env"
            )
    # ========================================================
    # HEADERS
    # ========================================================
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization":
                f"Bearer {self.api_key}",
            "Content-Type":
                "application/json",
        }
    # ========================================================
    # COMPLETE
    # ========================================================
    def complete(
        self,
        system: str,
        user: str,
        max_output_tokens: int | None = None,
    ) -> str:
        self._check_api_key()
        output_tokens = int(
            max_output_tokens
            if max_output_tokens is not None
            else self.max_output_tokens
        )
        output_tokens = max(
            1,
            output_tokens,
        )
        original_user = user
        estimated_input = estimate_tokens(
            system
        ) + estimate_tokens(
            user
        )
        # ----------------------------------------------------
        # Shrink oversized requests
        # ----------------------------------------------------
        if (
            estimated_input
            + output_tokens
            > self.max_request_tokens
        ):
            allowed_input = max(
                500,
                self.max_request_tokens
                - output_tokens,
            )
            user = self._shrink_text(
                user,
                allowed_input,
            )
            estimated_input = (
                estimate_tokens(system)
                + estimate_tokens(user)
            )
        if (
            estimated_input
            + output_tokens
            > self.max_request_tokens
        ):
            # Reduce output budget as a second protection.
            output_tokens = min(
                output_tokens,
                max(
                    200,
                    self.max_request_tokens
                    - estimated_input,
                ),
            )
        total_estimated = (
            estimated_input
            + output_tokens
        )
        if (
            total_estimated
            > self.max_request_tokens
        ):
            raise RequestTooLarge(
                "Request is too large for the "
                f"configured limit. "
                f"Estimated tokens: "
                f"{total_estimated}; "
                f"maximum: "
                f"{self.max_request_tokens}."
            )
        # ----------------------------------------------------
        # Retry loop
        # ----------------------------------------------------
        last_error = None
        for attempt in range(
            self.max_retries + 1
        ):
            try:
                self.limiter.wait_for_capacity(
                    total_estimated
                )
                self.limiter.reserve(
                    total_estimated
                )
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens":
                        output_tokens,
                }
                response = self.session.post(
                    self.url,
                    headers=self._headers(),
                    json=payload,
                    timeout=90,
                )
                # ====================================================
                # SUCCESS
                # ====================================================
                if response.ok:
                    try:
                        data = response.json()
                    except ValueError:
                        raise RuntimeError(
                            "Groq returned a non-JSON "
                            "response:\n"
                            + response.text[:1000]
                        )
                    choices = data.get(
                        "choices"
                    )
                    if not choices:
                        raise RuntimeError(
                            "Groq returned no choices.\n"
                            f"Response: {data}"
                        )
                    message = choices[0].get(
                        "message",
                        {},
                    )
                    content = message.get(
                        "content"
                    )
                    if content is None:
                        raise RuntimeError(
                            "Groq returned an empty "
                            "message.\n"
                            f"Response: {data}"
                        )
                    return str(
                        content
                    ).strip()
                # ====================================================
                # AUTHENTICATION
                # ====================================================
                if response.status_code in (
                    401,
                    403,
                ):
                    raise RuntimeError(
                        "Groq authentication failed "
                        f"(HTTP {response.status_code}).\n\n"
                        "Check GROQ_API_KEY in "
                        "Streamlit Secrets.\n\n"
                        f"Groq response: "
                        f"{response.text[:1000]}"
                    )
                # ====================================================
                # BAD REQUEST
                # ====================================================
                if response.status_code == 400:
                    error_text = (
                        response.text[:1500]
                    )
                    # Try shrinking once if Groq says
                    # the request is invalid because of size.
                    if (
                        "token" in error_text.lower()
                        or "context" in error_text.lower()
                        or "length" in error_text.lower()
                    ):
                        user = self._shrink_text(
                            user,
                            max(
                                500,
                                int(
                                    estimate_tokens(
                                        original_user
                                    )
                                    * 0.65
                                ),
                            ),
                        )
                        output_tokens = max(
                            200,
                            int(
                                output_tokens
                                * 0.8
                            ),
                        )
                        total_estimated = (
                            estimate_tokens(system)
                            + estimate_tokens(user)
                            + output_tokens
                        )
                        if attempt < self.max_retries:
                            time.sleep(
                                1.0 + attempt
                            )
                            continue
                    raise RuntimeError(
                        "Groq rejected the request "
                        "(HTTP 400).\n\n"
                        f"Response: {error_text}"
                    )
                # ====================================================
                # NOT FOUND
                # ====================================================
                if response.status_code == 404:
                    raise RuntimeError(
                        "Groq endpoint or model was "
                        "not found (HTTP 404).\n\n"
                        f"Model: {self.model}\n"
                        f"URL: {self.url}\n\n"
                        f"Groq response: "
                        f"{response.text[:1000]}"
                    )
                # ====================================================
                # REQUEST TOO LARGE
                # ====================================================
                if response.status_code == 413:
                    user = self._shrink_text(
                        user,
                        max(
                            500,
                            int(
                                estimate_tokens(
                                    user
                                )
                                * 0.60
                            ),
                        ),
                    )
                    output_tokens = max(
                        200,
                        int(
                            output_tokens
                            * 0.75
                        ),
                    )
                    total_estimated = (
                        estimate_tokens(system)
                        + estimate_tokens(user)
                        + output_tokens
                    )
                    if attempt < self.max_retries:
                        time.sleep(
                            1.0 + attempt
                        )
                        continue
                    raise RequestTooLarge(
                        "Groq rejected the request "
                        "because it was too large "
                        "(HTTP 413)."
                    )
                # ====================================================
                # RATE LIMIT
                # ====================================================
                if response.status_code == 429:
                    retry_after = (
                        response.headers.get(
                            "retry-after"
                        )
                    )
                    if retry_after:
                        try:
                            wait_seconds = float(
                                retry_after
                            )
                        except ValueError:
                            wait_seconds = (
                                2.0
                                * (
                                    attempt + 1
                                )
                            )
                    else:
                        wait_seconds = (
                            2.0
                            * (
                                attempt + 1
                            )
                        )
                    if attempt < self.max_retries:
                        time.sleep(
                            min(
                                wait_seconds,
                                30.0,
                            )
                        )
                        continue
                    raise RateLimitExceeded(
                        "Groq rate limit exceeded "
                        "(HTTP 429).\n\n"
                        f"Groq response: "
                        f"{response.text[:1000]}"
                    )
                # ====================================================
                # SERVER ERRORS
                # ====================================================
                if response.status_code >= 500:
                    last_error = RuntimeError(
                        "Groq server error "
                        f"(HTTP {response.status_code}).\n"
                        f"Response: "
                        f"{response.text[:1000]}"
                    )
                    if attempt < self.max_retries:
                        time.sleep(
                            min(
                                2 ** attempt,
                                10,
                            )
                        )
                        continue
                    raise last_error
                # ====================================================
                # OTHER HTTP ERRORS
                # ====================================================
                raise RuntimeError(
                    "Groq API error "
                    f"(HTTP {response.status_code}).\n\n"
                    f"Response: "
                    f"{response.text[:1000]}"
                )
            except requests.Timeout as exc:
                last_error = RuntimeError(
                    "Groq request timed out."
                )
                if attempt < self.max_retries:
                    time.sleep(
                        1.5
                        * (
                            attempt + 1
                        )
                    )
                    continue
                raise last_error from exc
            except requests.ConnectionError as exc:
                last_error = RuntimeError(
                    "Could not connect to Groq."
                )
                if attempt < self.max_retries:
                    time.sleep(
                        1.5
                        * (
                            attempt + 1
                        )
                    )
                    continue
                raise last_error from exc
            except requests.RequestException as exc:
                last_error = RuntimeError(
                    "Network error while calling Groq: "
                    f"{exc}"
                )
                if attempt < self.max_retries:
                    time.sleep(
                        1.5
                        * (
                            attempt + 1
                        )
                    )
                    continue
                raise last_error from exc
        if last_error:
            raise last_error
        raise RuntimeError(
            "Groq request failed for an unknown reason."
        )
    # ========================================================
    # JSON COMPLETE
    # ========================================================
    def json_complete(
        self,
        system: str,
        user: str,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        json_instruction = """
Return valid JSON only.
Do not use Markdown.
Do not use ```json fences.
Do not add explanations before or after the JSON.
The response must be a single valid JSON object.
"""
        combined_system = (
            system
            + "\n"
            + json_instruction
        )
        text = self.complete(
            combined_system,
            user,
            max_output_tokens=(
                max_output_tokens
                or 700
            ),
        )
        cleaned = text.strip()
        # ----------------------------------------------------
        # Remove accidental Markdown fences
        # ----------------------------------------------------
        if cleaned.startswith(
            "```json"
        ):
            cleaned = cleaned[
                len("```json"):
            ].strip()
        elif cleaned.startswith(
            "```"
        ):
            cleaned = cleaned[
                len("```"):
            ].strip()
        if cleaned.endswith(
            "```"
        ):
            cleaned = cleaned[
                :-3
            ].strip()
        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------
        try:
            result = json.loads(
                cleaned
            )
        except json.JSONDecodeError as exc:
            # Try extracting the outermost JSON object.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if (
                start >= 0
                and end > start
            ):
                candidate = cleaned[
                    start:end + 1
                ]
                try:
                    result = json.loads(
                        candidate
                    )
                except json.JSONDecodeError:
                    raise RuntimeError(
                        "Groq returned invalid JSON.\n\n"
                        f"Raw response:\n{cleaned[:3000]}"
                    ) from exc
            else:
                raise RuntimeError(
                    "Groq returned invalid JSON.\n\n"
                    f"Raw response:\n{cleaned[:3000]}"
                ) from exc
        if not isinstance(
            result,
            dict,
        ):
            raise RuntimeError(
                "Groq JSON response was not "
                "a JSON object."
            )
        return result
    # ========================================================
    # SHRINK TEXT
    # ========================================================
    def _shrink_text(
        self,
        text: str,
        target_tokens: int,
    ) -> str:
        if not text:
            return text
        target_tokens = max(
            100,
            int(target_tokens),
        )
        target_chars = (
            target_tokens * 4
        )
        if len(text) <= target_chars:
            return text
        # Keep both beginning and end.
        # This is safer for JSON-like prompts because
        # useful information may exist at either side.
        head_chars = int(
            target_chars * 0.75
        )
        tail_chars = (
            target_chars
            - head_chars
        )
        return (
            text[:head_chars]
            + "\n...[content shortened]...\n"
            + text[-tail_chars:]
        )
# ============================================================
# DEFAULT CLIENT
# ============================================================
client = GroqClient()
# ============================================================
# MODULE-LEVEL HELPERS
# ============================================================
def complete(
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> str:
    return client.complete(
        system,
        user,
        max_output_tokens=max_output_tokens,
    )
def json_complete(
    system: str,
    user: str,
    max_output_tokens: int = DEFAULT_OUTPUT_TOKENS,
) -> dict[str, Any]:
    return client.json_complete(
        system,
        user,
        max_output_tokens=max_output_tokens,
    )
# ============================================================
# CHUNK RECORDS
# ============================================================
def chunk_records(
    records: Iterable[Any],
    max_tokens: int = 3000,
) -> list[list[Any]]:
    """
    Split records into token-safe batches.
    """
    batches: list[list[Any]] = []
    current: list[Any] = []
    current_tokens = 0
    for record in records:
        record_text = compact_json(
            record
        )
        record_tokens = estimate_tokens(
            record_text
        )
        if (
            current
            and current_tokens
            + record_tokens
            > max_tokens
        ):
            batches.append(
                current
            )
            current = []
            current_tokens = 0
        current.append(
            record
        )
        current_tokens += (
            record_tokens
        )
    if current:
        batches.append(
            current
        )
    return batches

Then make sure your files are exactly

student-performance-agent-groq/
│
├── app.py
├── agents.py
├── llm_gateway.py
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── calculations.py
│   └── validation.py
│
└── storage/
    ├── __init__.py
    └── database.py

And your requirements.txt should at least contain:

streamlit
pandas
requests
python-dotenv
openpyxl

Do not add tiktoken just for this fix. The new gateway deliberately uses a lightweight token estimate.

After replacing the two files, restart/redeploy the Streamlit app. If it fails again, the new gateway should show the actual Groq HTTP status and response instead of the old redacted _prune() traceback.
