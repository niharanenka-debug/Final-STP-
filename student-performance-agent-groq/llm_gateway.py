from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


def config_value(name: str, default: str = "") -> str:
    """Read local .env first, then Streamlit Cloud Secrets."""
    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:
        return default


try:
    import tiktoken
except ImportError:
    tiktoken = None


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

TPM_LIMIT = int(config_value("GROQ_TPM_LIMIT", "8000"))
WINDOW_SECONDS = 60.0

TARGET_INPUT_TOKENS = int(
    config_value("GROQ_CHUNK_INPUT_TOKENS", "4000")
)

MAX_REQUEST_TOKENS = int(
    config_value("GROQ_MAX_REQUEST_TOKENS", "6000")
)

DEFAULT_OUTPUT_TOKENS = int(
    config_value("GROQ_MAX_OUTPUT_TOKENS", "700")
)

MAX_RETRIES = int(
    config_value("GROQ_MAX_RETRIES", "4")
)


# -------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------

class RequestTooLarge(RuntimeError):
    """Raised when a request is too large."""


class RateLimitExceeded(RuntimeError):
    """Raised when a request cannot be completed after retries."""


# -------------------------------------------------------------------
# Token helpers
# -------------------------------------------------------------------

def estimate_tokens(value: Any) -> int:
    """Estimate token count."""

    if not isinstance(value, str):
        value = json.dumps(
            value,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )

    if tiktoken is not None:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            return max(1, len(encoder.encode(value)))
        except Exception:
            pass

    # Fallback approximation
    return max(1, (len(value) + 3) // 4)


def compact_json(value: Any) -> str:
    """Convert Python object to compact JSON."""
    return json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


# -------------------------------------------------------------------
# Usage tracking
# -------------------------------------------------------------------

@dataclass
class UsageEvent:
    timestamp: float
    estimated_tokens: int
    actual_tokens: int | None = None

    @property
    def reserved_tokens(self) -> int:
        return max(
            self.estimated_tokens,
            self.actual_tokens or 0,
        )


class RollingTokenLimiter:
    """Simple rolling-window token limiter."""

    def __init__(
        self,
        limit: int = TPM_LIMIT,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: deque[UsageEvent] = deque()
        self.lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while (
            self.events
            and now - self.events[0].timestamp
            >= self.window_seconds
        ):
            self.events.popleft()

    def used(self) -> int:
        with self.lock:
            now = time.monotonic()
            self._prune(now)

            return sum(
                event.reserved_tokens
                for event in self.events
            )

    def wait_for_budget(self, tokens: int) -> None:
        if tokens > self.limit:
            raise RateLimitExceeded(
                f"Request needs approximately {tokens} tokens, "
                f"but configured TPM limit is {self.limit}."
            )

        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)

                used = sum(
                    event.reserved_tokens
                    for event in self.events
                )

                if used + tokens <= self.limit:
                    return

                sleep_for = max(
                    0.25,
                    self.window_seconds
                    - (now - self.events[0].timestamp)
                    + 0.05,
                )

            time.sleep(sleep_for)

    def reserve(
        self,
        estimated_tokens: int,
    ) -> UsageEvent:

        self.wait_for_budget(estimated_tokens)

        event = UsageEvent(
            timestamp=time.monotonic(),
            estimated_tokens=estimated_tokens,
        )

        with self.lock:
            self.events.append(event)

        return event

    def record_actual(
        self,
        event: UsageEvent,
        actual_tokens: int | None,
    ) -> None:

        if actual_tokens is not None:
            event.actual_tokens = actual_tokens


# -------------------------------------------------------------------
# Groq client
# -------------------------------------------------------------------

class GroqClient:

    def __init__(self) -> None:

        self.api_key = (
            config_value("GROQ_API_KEY")
            or config_value("LLM_API_KEY")
        )

        self.model = config_value(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        )

        self.url = config_value(
            "GROQ_API_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        )

        self.max_output_tokens = int(
            config_value(
                "GROQ_MAX_OUTPUT_TOKENS",
                str(DEFAULT_OUTPUT_TOKENS),
            )
        )

        self.max_request_tokens = int(
            config_value(
                "GROQ_MAX_REQUEST_TOKENS",
                str(MAX_REQUEST_TOKENS),
            )
        )

        self.limiter = RollingTokenLimiter()

        self.session = requests.Session()

    # ---------------------------------------------------------------
    # Main completion method
    # ---------------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_output_tokens: int | None = None,
    ) -> str:

        # -----------------------------------------------------------
        # API key check
        # -----------------------------------------------------------

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. "
                "Add GROQ_API_KEY to Streamlit Cloud Secrets "
                "or your local .env file."
            )

        output_budget = min(
            max_output_tokens or self.max_output_tokens,
            self.max_output_tokens,
        )

        current_user = user
        last_error: Exception | None = None

        # -----------------------------------------------------------
        # Retry loop
        # -----------------------------------------------------------

        for attempt in range(MAX_RETRIES + 1):

            input_tokens = (
                estimate_tokens(system)
                + estimate_tokens(current_user)
            )

            reserved = input_tokens + output_budget

            # -------------------------------------------------------
            # Request too large
            # -------------------------------------------------------

            if reserved > self.max_request_tokens:

                if attempt >= MAX_RETRIES:
                    raise RequestTooLarge(
                        f"Estimated request is {reserved} tokens, "
                        f"but maximum configured request size is "
                        f"{self.max_request_tokens}."
                    )

                current_user = self._shrink_text(
                    current_user,
                    0.55,
                )

                output_budget = max(
                    250,
                    int(output_budget * 0.75),
                )

                continue

            # -------------------------------------------------------
            # Reserve token budget
            # -------------------------------------------------------

            event = self.limiter.reserve(reserved)

            try:

                # ---------------------------------------------------
                # Groq API request
                # ---------------------------------------------------

                response = self.session.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "temperature": 0.1,
                        "max_tokens": output_budget,
                        "messages": [
                            {
                                "role": "system",
                                "content": system,
                            },
                            {
                                "role": "user",
                                "content": current_user,
                            },
                        ],
                    },
                    timeout=120,
                )

                # ---------------------------------------------------
                # SUCCESS
                # ---------------------------------------------------

                if response.ok:

                    try:
                        data = response.json()
                    except ValueError as exc:
                        raise RuntimeError(
                            "Groq returned a non-JSON response:\n"
                            + response.text[:1000]
                        ) from exc

                    usage = data.get("usage", {})

                    actual_tokens = (
                        (usage.get("prompt_tokens") or 0)
                        + (usage.get("completion_tokens") or 0)
                    )

                    self.limiter.record_actual(
                        event,
                        actual_tokens or None,
                    )

                    choices = data.get("choices", [])

                    if not choices:
                        raise RuntimeError(
                            "Groq returned no choices.\n"
                            f"Response: {data}"
                        )

                    message = choices[0].get(
                        "message",
                        {},
                    )

                    content = message.get("content")

                    if not content:
                        raise RuntimeError(
                            "Groq returned an empty response.\n"
                            f"Response: {data}"
                        )

                    return content

                # ---------------------------------------------------
                # ERROR DETAILS
                # ---------------------------------------------------

                status = response.status_code
                body = response.text[:1500]

                last_error = RuntimeError(
                    f"Groq API error {status}: {body}"
                )

                # ---------------------------------------------------
                # Authentication
                # ---------------------------------------------------

                if status == 401:
                    raise RuntimeError(
                        "Groq authentication failed (HTTP 401).\n\n"
                        "Check your GROQ_API_KEY in Streamlit "
                        "Cloud Secrets.\n\n"
                        f"Provider response:\n{body}"
                    )

                # ---------------------------------------------------
                # Forbidden
                # ---------------------------------------------------

                if status == 403:
                    raise RuntimeError(
                        "Groq rejected the request (HTTP 403).\n\n"
                        f"Provider response:\n{body}"
                    )

                # ---------------------------------------------------
                # Bad request
                # ---------------------------------------------------

                if status == 400:
                    raise RuntimeError(
                        "Groq rejected the request (HTTP 400).\n\n"
                        f"Model: {self.model}\n"
                        f"Provider response:\n{body}"
                    )

                # ---------------------------------------------------
                # Model not found
                # ---------------------------------------------------

                if status == 404:
                    raise RuntimeError(
                        "Groq endpoint or model was not found "
                        "(HTTP 404).\n\n"
                        f"Model: {self.model}\n"
                        f"URL: {self.url}\n\n"
                        f"Provider response:\n{body}"
                    )

                # ---------------------------------------------------
                # Validation error
                # ---------------------------------------------------

                if status == 422:
                    raise RuntimeError(
                        "Groq rejected the request parameters "
                        "(HTTP 422).\n\n"
                        f"Provider response:\n{body}"
                    )

                # ---------------------------------------------------
                # Request too large
                # ---------------------------------------------------

                if status == 413:

                    if attempt >= MAX_RETRIES:
                        raise RequestTooLarge(
                            f"Groq rejected the request as too large.\n"
                            f"Response: {body}"
                        )

                    current_user = self._shrink_text(
                        current_user,
                        0.55,
                    )

                    output_budget = max(
                        250,
                        int(output_budget * 0.75),
                    )

                    continue

                # ---------------------------------------------------
                # Rate limit
                # ---------------------------------------------------

                if status == 429:

                    if attempt >= MAX_RETRIES:
                        raise RateLimitExceeded(
                            "Groq rate limit exceeded.\n\n"
                            f"Provider response:\n{body}"
                        )

                    retry_after = response.headers.get(
                        "retry-after",
                        "0",
                    )

                    try:
                        retry_delay = float(
                            retry_after
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        retry_delay = 0.0

                    wait_time = max(
                        retry_delay,
                        1.0,
                        self._wait_until_next_window(),
                    )

                    time.sleep(wait_time)

                    current_user = self._shrink_text(
                        current_user,
                        0.75,
                    )

                    output_budget = max(
                        250,
                        int(output_budget * 0.85),
                    )

                    continue

                # ---------------------------------------------------
                # Server errors
                # ---------------------------------------------------

                if status >= 500:

                    if attempt >= MAX_RETRIES:
                        raise RuntimeError(
                            "Groq server error after retries.\n\n"
                            f"HTTP {status}\n"
                            f"Provider response:\n{body}"
                        )

                    time.sleep(
                        min(2 ** attempt, 8)
                    )

                    continue

                # ---------------------------------------------------
                # Unknown HTTP error
                # ---------------------------------------------------

                raise RuntimeError(
                    f"Groq API error {status}:\n{body}"
                )

            except requests.RequestException as exc:

                last_error = exc

                if attempt >= MAX_RETRIES:
                    break

                time.sleep(
                    min(2 ** attempt, 8)
                )

        # -----------------------------------------------------------
        # Final failure
        # -----------------------------------------------------------

        raise RuntimeError(
            "Groq request failed after retries.\n\n"
            f"Last error: {last_error}"
        ) from last_error

    # ---------------------------------------------------------------
    # JSON completion
    # ---------------------------------------------------------------

    def json_complete(
        self,
        system: str,
        user: str,
        *,
        max_output_tokens: int = 500,
    ) -> dict[str, Any]:

        text = self.complete(
            system
            + " Return valid compact JSON only. "
              "No markdown fences. Keep arrays short.",
            user,
            max_output_tokens=max_output_tokens,
        )

        text = text.strip()

        # Remove markdown fences if model adds them.
        if text.startswith("```json"):
            text = text[7:]

        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            result = json.loads(text)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Groq returned invalid JSON.\n\n"
                f"Raw response:\n{text[:1500]}"
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                "Groq returned valid JSON, but it was not "
                "a JSON object.\n\n"
                f"Response:\n{text[:1500]}"
            )

        return result

    # ---------------------------------------------------------------
    # Token window
    # ---------------------------------------------------------------

    def _wait_until_next_window(self) -> float:

        with self.limiter.lock:

            if not self.limiter.events:
                return 0.0

            remaining = (
                WINDOW_SECONDS
                - (
                    time.monotonic()
                    - self.limiter.events[0].timestamp
                )
            )

            return max(
                0.0,
                remaining + 0.05,
            )

    # ---------------------------------------------------------------
    # Shrink oversized input
    # ---------------------------------------------------------------

    @staticmethod
    def _shrink_text(
        text: str,
        ratio: float,
    ) -> str:

        target_chars = max(
            1000,
            int(len(text) * ratio),
        )

        return (
            text[:target_chars]
            + "\n"
            "[Input compacted after provider limit; "
            "use only the retained records.]"
        )


# -------------------------------------------------------------------
# Record chunking
# -------------------------------------------------------------------

def chunk_records(
    records: list[dict[str, Any]],
    system_prompt: str,
    *,
    target_tokens: int = TARGET_INPUT_TOKENS,
) -> list[list[dict[str, Any]]]:

    """Pack records into token-limited chunks."""

    chunks: list[list[dict[str, Any]]] = []

    current: list[dict[str, Any]] = []

    base_tokens = estimate_tokens(
        system_prompt
    )

    used = base_tokens

    for record in records:

        record_tokens = (
            estimate_tokens(
                compact_json(record)
            )
            + 2
        )

        if (
            current
            and used + record_tokens > target_tokens
        ):
            chunks.append(current)

            current = []

            used = base_tokens

        current.append(record)

        used += record_tokens

    if current:
        chunks.append(current)

    return chunks
