from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

load_dotenv()


def config_value(name: str, default: str = "") -> str:
    """Read local .env first, then Streamlit Cloud Secrets when deployed."""
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
except ImportError:  # pragma: no cover - optional dependency fallback
    tiktoken = None


TPM_LIMIT = int(config_value("GROQ_TPM_LIMIT", "8000"))
WINDOW_SECONDS = 60.0
TARGET_INPUT_TOKENS = int(config_value("GROQ_CHUNK_INPUT_TOKENS", "4000"))
MAX_REQUEST_TOKENS = int(config_value("GROQ_MAX_REQUEST_TOKENS", "6000"))
DEFAULT_OUTPUT_TOKENS = int(config_value("GROQ_MAX_OUTPUT_TOKENS", "700"))
MAX_RETRIES = int(config_value("GROQ_MAX_RETRIES", "4"))


class RequestTooLarge(RuntimeError):
    """Raised when a request is rejected as too large after adaptive retries."""


class RateLimitExceeded(RuntimeError):
    """Raised when a request cannot be completed within the configured retry budget."""


def estimate_tokens(value: Any) -> int:
    """Estimate tokens before a request. Uses tiktoken when available, chars otherwise."""
    if not isinstance(value, str):
        value = json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
    if tiktoken is not None:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
            return max(1, len(encoder.encode(value)))
        except Exception:
            pass
    return max(1, (len(value) + 3) // 4)


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass
class UsageEvent:
    timestamp: float
    estimated_tokens: int
    actual_tokens: int | None = None

    @property
    def reserved_tokens(self) -> int:
        return max(self.estimated_tokens, self.actual_tokens or 0)


class RollingTokenLimiter:
    """Sequential rolling-window limiter. It reserves input + output tokens."""

    def __init__(self, limit: int = TPM_LIMIT, window_seconds: float = WINDOW_SECONDS) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.events: deque[UsageEvent] = deque()
        self.lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self.events and now - self.events[0].timestamp >= self.window_seconds:
            self.events.popleft()

    def used(self) -> int:
        with self.lock:
            now = time.monotonic()
            self._prune(now)
            return sum(event.reserved_tokens for event in self.events)

    def wait_for_budget(self, tokens: int) -> None:
        if tokens > self.limit:
            raise RateLimitExceeded(f"One request reserves {tokens} tokens, above the {self.limit} TPM limit.")
        while True:
            with self.lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(event.reserved_tokens for event in self.events)
                if used + tokens <= self.limit:
                    return
                sleep_for = max(0.25, self.window_seconds - (now - self.events[0].timestamp) + 0.05)
            time.sleep(sleep_for)

    def reserve(self, estimated_tokens: int) -> UsageEvent:
        self.wait_for_budget(estimated_tokens)
        event = UsageEvent(time.monotonic(), estimated_tokens)
        with self.lock:
            self.events.append(event)
        return event

    def record_actual(self, event: UsageEvent, actual_tokens: int | None) -> None:
        if actual_tokens is None:
            return
        event.actual_tokens = actual_tokens


class GroqClient:
    def __init__(self) -> None:
        self.api_key = config_value("GROQ_API_KEY") or config_value("LLM_API_KEY")
        self.model = config_value("GROQ_MODEL", "openai/gpt-oss-20b")
        self.url = config_value("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.max_output_tokens = int(config_value("GROQ_MAX_OUTPUT_TOKENS", str(DEFAULT_OUTPUT_TOKENS)))
        self.max_request_tokens = int(config_value("GROQ_MAX_REQUEST_TOKENS", str(MAX_REQUEST_TOKENS)))
        self.limiter = RollingTokenLimiter()
        self.session = requests.Session()

    def complete(self, system: str, user: str, *, max_output_tokens: int | None = None) -> str:
        output_budget = min(max_output_tokens or self.max_output_tokens, self.max_output_tokens)
        current_user = user
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            input_tokens = estimate_tokens(system) + estimate_tokens(current_user)
            reserved = input_tokens + output_budget
            if reserved > self.max_request_tokens:
                if attempt >= MAX_RETRIES:
                    raise RequestTooLarge(f"Estimated request is {reserved} tokens after compaction; reduce the input before sending.")
                current_user = self._shrink_text(current_user, 0.55)
                output_budget = max(250, int(output_budget * 0.75))
                continue
            event = self.limiter.reserve(reserved)
            try:
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
                            {"role": "system", "content": system},
                            {"role": "user", "content": current_user},
                        ],
                    },
                    timeout=120,
                )
                if response.status_code not in (429, 413):
                    if not response.ok:
                        raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:500]}")
                    data = response.json()
                    usage = data.get("usage", {})
                    actual = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
                    self.limiter.record_actual(event, actual or None)
                    return data["choices"][0]["message"]["content"]

                last_error = RuntimeError(f"Groq HTTP {response.status_code}: {response.text[:300]}")
                if attempt >= MAX_RETRIES:
                    break
                # Do not repeat the same request. Reserve a smaller retry after the window wait.
                retry_after = float(response.headers.get("retry-after", "0") or 0)
                time.sleep(max(retry_after, 1.0, self._wait_until_next_window()))
                current_user = self._shrink_text(current_user, 0.55)
                output_budget = max(250, int(output_budget * 0.75))
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= MAX_RETRIES:
                    break
                time.sleep(min(2 ** attempt, 8))
        if isinstance(last_error, RuntimeError) and "413" in str(last_error):
            raise RequestTooLarge(str(last_error))
        raise RateLimitExceeded(str(last_error or "Groq request failed"))

    def json_complete(self, system: str, user: str, *, max_output_tokens: int = 500) -> dict[str, Any]:
        text = self.complete(
            system + " Return valid compact JSON only. No markdown fences. Keep arrays short.",
            user,
            max_output_tokens=max_output_tokens,
        )
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Groq returned invalid JSON: {text[:500]}") from exc

    def _wait_until_next_window(self) -> float:
        with self.limiter.lock:
            if not self.limiter.events:
                return 0.0
            remaining = WINDOW_SECONDS - (time.monotonic() - self.limiter.events[0].timestamp)
            return max(0.0, remaining + 0.05)

    @staticmethod
    def _shrink_text(text: str, ratio: float) -> str:
        target_chars = max(1000, int(len(text) * ratio))
        return text[:target_chars] + "\n[Input compacted after provider limit; use only the retained records.]"


def chunk_records(records: list[dict[str, Any]], system_prompt: str, *, target_tokens: int = TARGET_INPUT_TOKENS) -> list[list[dict[str, Any]]]:
    """Pack records sequentially without exceeding the target estimated input size."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    base_tokens = estimate_tokens(system_prompt)
    used = base_tokens
    for record in records:
        record_tokens = estimate_tokens(compact_json(record)) + 2
        if current and used + record_tokens > target_tokens:
            chunks.append(current)
            current = []
            used = base_tokens
        current.append(record)
        used += record_tokens
    if current:
        chunks.append(current)
    return chunks
