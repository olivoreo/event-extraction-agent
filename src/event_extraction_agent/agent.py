from __future__ import annotations

import json
import re
import threading
import time as time_module
import urllib.error
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Protocol

from event_extraction_agent.models import (
    Event,
    ExtractionError,
    ExtractionOutcome,
    ExtractionStatus,
    SourcePost,
)
from event_extraction_agent.prompts import (
    ATTENDANCE_TYPE_VALUES,
    EVENT_STATUS_VALUES,
    EVENT_TYPE_CLASSIFICATION_PROMPT,
    EVENT_TYPE_VALUES,
    INDUSTRY_VALUES,
    ROLE_VALUES,
    SKIP_REASONS,
    SYSTEM_PROMPT,
    build_event_type_classification_prompt,
    build_extraction_prompt,
)
from event_extraction_agent.validator import ValidationIssue, validate_extraction_result

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MAIN_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_GROQ_MAX_RETRIES = 3

_MONTHS = {
    "января": 1,
    "январь": 1,
    "февраля": 2,
    "февраль": 2,
    "марта": 3,
    "март": 3,
    "апреля": 4,
    "апрель": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "сентября": 9,
    "сентябрь": 9,
    "октября": 10,
    "октябрь": 10,
    "ноября": 11,
    "ноябрь": 11,
    "декабря": 12,
    "декабрь": 12,
}
_MONTH_PATTERN = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_TIME_PATTERN = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s*(?P<month>{_MONTH_PATTERN})\b"
    r"(?:(?:(?!\b\d{1,2}\s*(?:"
    + _MONTH_PATTERN
    + r")\b).){0,80}?)"
    r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b",
    re.IGNORECASE | re.UNICODE | re.DOTALL,
)
_LOCAL_CONTEXT_WORDS = re.compile(
    r"\b(волгоград|москва|санкт-петербург|амфитеатр|ресторан|храм|набережная)\b",
    re.IGNORECASE | re.UNICODE,
)


class LLMClient(Protocol):
    """Minimal interface used by ExtractionAgent."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return model content as text."""


class OllamaChatClient:
    """Small Ollama `/api/chat` client using only the Python standard library."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            detail = f" HTTP {exc.code}: {error_body}" if error_body else f" HTTP {exc.code}"
            raise RuntimeError(f"Ollama request failed:{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            response_payload = json.loads(body)
            return str(response_payload["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Ollama response does not contain message.content") from exc


class RequestRateLimiter:
    """Thread-safe minimum interval limiter for provider API calls."""

    def __init__(
        self,
        min_interval_seconds: float,
        sleep: Any = time_module.sleep,
        monotonic: Any = time_module.monotonic,
    ) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = self._monotonic()
            delay = self._next_allowed_at - now
            if delay > 0:
                self._sleep(delay)
                now = self._monotonic()
            self._next_allowed_at = now + self.min_interval_seconds


class GroqChatClient:
    """OpenAI-compatible Groq chat completions client."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MAIN_MODEL,
        base_url: str = DEFAULT_GROQ_BASE_URL,
        timeout_seconds: float = 120.0,
        rate_limiter: RequestRateLimiter | None = None,
        max_retries: int = DEFAULT_GROQ_MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise ValueError("Groq API key is required")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = rate_limiter
        self.max_retries = max(0, max_retries)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "event-extraction-agent/0.1",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        body = self._send_with_retries(request)

        try:
            response_payload = json.loads(body)
            return str(response_payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Groq response does not contain choices[0].message.content") from exc

    def _send_with_retries(self, request: urllib.request.Request) -> str:
        last_error: urllib.error.HTTPError | None = None
        for attempt in range(self.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                if not _is_retryable_http_error(exc) or attempt >= self.max_retries:
                    error_body = exc.read().decode("utf-8", errors="replace").strip()
                    detail = f" HTTP {exc.code}: {error_body}" if error_body else f" HTTP {exc.code}"
                    raise RuntimeError(f"Groq request failed:{detail}") from exc
                last_error = exc
                _sleep_before_retry(exc, attempt)
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Groq request failed: {exc}") from exc
        if last_error is not None:
            raise RuntimeError(f"Groq request failed: HTTP {last_error.code}") from last_error
        raise RuntimeError("Groq request failed")


class ExtractionAgent:
    """Extract one source post through an LLM and validate the resulting event."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        event_type_llm_client: LLMClient | None = None,
        current_datetime: str | None = None,
    ) -> None:
        self.llm_client = llm_client or OllamaChatClient()
        self.event_type_llm_client = event_type_llm_client or self.llm_client
        self.current_datetime = current_datetime
        self.last_metadata: dict[str, Any] = {}

    def extract(self, post: SourcePost) -> ExtractionOutcome:
        prompt = build_extraction_prompt(
            raw_text=post.text,
            source_name=post.source_name,
            source_url=post.source_url,
            published_at=post.published_at_for_prompt(),
            external_id=post.external_id,
            current_datetime=self.current_datetime or _current_datetime(),
        )
        self.last_metadata = {
            "llm_model": getattr(self.llm_client, "model", None),
            "external_id": post.external_id,
        }

        try:
            llm_content = self.llm_client.complete(SYSTEM_PROMPT, prompt)
            response_payload = _parse_llm_json(llm_content)
        except Exception as exc:
            return _outcome(
                status=ExtractionStatus.LLM_ERROR,
                post=post,
                errors=[_error("llm", "llm_error", str(exc))],
                metadata=self.last_metadata,
            )

        if response_payload.get("is_event") is False:
            reason = response_payload.get("skip_reason")
            if not isinstance(reason, str) or reason not in SKIP_REASONS:
                reason = "not_event_announcement"
            return _outcome(
                status=ExtractionStatus.SKIPPED,
                post=post,
                errors=[_error("root", reason, "post is not an event announcement")],
                metadata=self.last_metadata,
            )

        event_payload = response_payload.get("event")
        if not isinstance(event_payload, dict):
            return _outcome(
                status=ExtractionStatus.LLM_ERROR,
                post=post,
                errors=[_error("event", "invalid_llm_shape", "LLM response must contain event object")],
                metadata=self.last_metadata,
            )

        event_payload = _repair_event_payload(
            event_payload,
            raw_text=post.text,
            published_at=post.published_at_for_prompt(),
        )
        event_payload = self._refine_event_type(event_payload, raw_text=post.text)
        event_payload = _with_source_metadata(
            event_payload,
            raw_text=post.text,
            source_name=post.source_name,
            source_url=post.source_url,
        )
        validation = validate_extraction_result(event_payload)
        if not validation.is_valid:
            return _outcome(
                status=ExtractionStatus.INVALID,
                post=post,
                errors=[_issue_to_error(issue) for issue in validation.errors],
                metadata=self.last_metadata,
            )

        assert validation.event is not None
        return _outcome(
            status=ExtractionStatus.EXTRACTED,
            post=post,
            event=validation.event,
            metadata=self.last_metadata,
        )

    def extract_event(self, post: SourcePost) -> Event:
        outcome = self.extract(post)
        if outcome.status != ExtractionStatus.EXTRACTED or outcome.event is None:
            raise ValueError(f"event extraction failed: {outcome.status}: {outcome.errors}")
        return outcome.event

    def _refine_event_type(self, event_payload: dict[str, Any], raw_text: str) -> dict[str, Any]:
        copied = dict(event_payload)
        event_type = copied.get("event_type")
        if event_type in EVENT_TYPE_VALUES and event_type != "SocialEvent":
            return copied

        prompt = build_event_type_classification_prompt(raw_text=raw_text, draft=copied)
        try:
            content = self.event_type_llm_client.complete(EVENT_TYPE_CLASSIFICATION_PROMPT, prompt)
            payload = _parse_llm_json(content)
        except Exception as exc:
            self.last_metadata["event_type_refine_error"] = str(exc)
            if event_type not in EVENT_TYPE_VALUES:
                copied["event_type"] = "SocialEvent"
            return copied

        refined_event_type = payload.get("event_type")
        if refined_event_type in EVENT_TYPE_VALUES:
            copied["event_type"] = refined_event_type
            self.last_metadata["event_type_refined"] = True
        elif event_type not in EVENT_TYPE_VALUES:
            copied["event_type"] = "SocialEvent"
        return copied


def _parse_llm_json(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object")
    return payload


def _with_source_metadata(
    payload: dict[str, Any],
    raw_text: str,
    source_name: str | None,
    source_url: str | None,
) -> dict[str, Any]:
    copied = {key: value for key, value in payload.items() if key in _EVENT_FIELDS}
    copied["raw_text"] = raw_text
    copied["source_name"] = source_name
    copied["source_url"] = source_url
    return copied


def _repair_event_payload(payload: dict[str, Any], raw_text: str, published_at: str | None) -> dict[str, Any]:
    copied = {key: value for key, value in payload.items() if key in _EVENT_FIELDS}
    if not isinstance(copied.get("title"), str) or not copied.get("title", "").strip():
        copied["title"] = _infer_title(raw_text)
    if not isinstance(copied.get("language"), str) or not copied.get("language", "").strip():
        copied["language"] = "ru"
    if _is_missing_value(copied.get("timezone")) and _LOCAL_CONTEXT_WORDS.search(raw_text):
        copied["timezone"] = "Europe/Moscow"
    if _is_missing_value(copied.get("start_at")):
        inferred_start_at = _extract_start_at(raw_text, published_at)
        if inferred_start_at is not None:
            copied["start_at"] = inferred_start_at
    if _is_missing_value(copied.get("price_text")):
        copied["price_text"] = "free"
    _coerce_prompt_enum(copied, "attendance_type", ATTENDANCE_TYPE_VALUES)
    _coerce_prompt_enum(copied, "event_status", EVENT_STATUS_VALUES)
    _filter_prompt_list(copied, "relevant_roles", ROLE_VALUES)
    _filter_prompt_list(copied, "industries", INDUSTRY_VALUES)
    if copied.get("attendance_type") == "unknown":
        copied["attendance_type"] = "OfflineEventAttendanceMode"
    if copied.get("event_status") == "unknown":
        copied["event_status"] = "EventScheduled"
    return copied


def _extract_start_at(raw_text: str, published_at: str | None) -> str | None:
    match = _DATE_TIME_PATTERN.search(raw_text)
    if match is None:
        return None

    day = int(match.group("day"))
    month = _MONTHS[match.group("month").lower()]
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    if not (1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    published_date = _published_date(published_at)
    year = published_date.year if published_date is not None else datetime.now(timezone.utc).year
    try:
        event_date = date(year, month, day)
    except ValueError:
        return None
    if published_date is not None and event_date < published_date:
        try:
            event_date = date(year + 1, month, day)
        except ValueError:
            return None

    event_datetime = datetime.combine(event_date, time(hour, minute), tzinfo=timezone(timedelta(hours=3)))
    return event_datetime.isoformat()


def _published_date(published_at: str | None) -> date | None:
    if not published_at:
        return None
    try:
        normalized = published_at.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        return None


def _infer_title(raw_text: str) -> str:
    first_sentence = re.split(r"[.!?]\s+", raw_text, maxsplit=1)[0]
    title = first_sentence.strip(" -|")
    if len(title) > 120:
        title = title[:120].rsplit(" ", 1)[0]
    return title or "Событие"


def _is_missing_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "unknown", "null"})


def _coerce_prompt_enum(payload: dict[str, Any], field: str, allowed_values: tuple[str, ...]) -> None:
    value = payload.get(field)
    if value not in allowed_values:
        payload[field] = "unknown"


def _filter_prompt_list(payload: dict[str, Any], field: str, allowed_values: tuple[str, ...]) -> None:
    value = payload.get(field)
    if value is None:
        return
    if not isinstance(value, list):
        payload[field] = None
        return

    allowed = set(allowed_values)
    filtered = [item for item in value if isinstance(item, str) and item in allowed]
    payload[field] = filtered or None


def _is_retryable_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code == 429 or 500 <= exc.code <= 599


def _sleep_before_retry(exc: urllib.error.HTTPError, attempt: int) -> None:
    retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
    if retry_after:
        try:
            delay = max(0.0, float(retry_after))
        except ValueError:
            delay = 0.0
    else:
        delay = min(30.0, 2.0**attempt)
    if delay > 0:
        time_module.sleep(delay)


def _current_datetime() -> str:
    return datetime.now(timezone(timedelta(hours=3))).isoformat()


def _outcome(
    status: ExtractionStatus,
    post: SourcePost,
    event: Event | None = None,
    errors: list[ExtractionError] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExtractionOutcome:
    return ExtractionOutcome(
        status=status,
        event=event,
        post=post,
        errors=errors or [],
        raw_llm_metadata=metadata or None,
    )


def _issue_to_error(issue: ValidationIssue) -> ExtractionError:
    return _error(issue.field, issue.code, issue.message)


def _error(field: str, code: str, message: str) -> ExtractionError:
    return ExtractionError(field=field, code=code, message=message)


_EVENT_FIELDS = {
    "title",
    "description",
    "start_at",
    "end_at",
    "timezone",
    "city",
    "venue_name",
    "address",
    "event_type",
    "attendance_type",
    "event_status",
    "language",
    "source_name",
    "source_url",
    "raw_text",
    "relevant_roles",
    "industries",
    "skills",
    "price_text",
    "target_audience_text",
}
