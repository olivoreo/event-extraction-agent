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
    BatchExtractionResult,
    BatchExtractionSettings,
    Event,
    ExtractionAgentConfig,
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
_DATE_RANGE_TIME_PATTERN = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s*(?:и|[-–—])\s*\d{{1,2}}\s*(?P<month>{_MONTH_PATTERN})\b"
    r"(?:(?:(?!\b\d{1,2}\s*(?:"
    + _MONTH_PATTERN
    + r")\b).){0,80}?)"
    r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b",
    re.IGNORECASE | re.UNICODE | re.DOTALL,
)
_LISTED_DATES_TIME_PATTERN = re.compile(
    rf"\b(?P<days>\d{{1,2}}(?:\s*(?:,|и)\s*\d{{1,2}})+)\s*(?P<month>{_MONTH_PATTERN})\b"
    r"(?:(?:(?!\b\d{1,2}\s*(?:"
    + _MONTH_PATTERN
    + r")\b).){0,80}?)"
    r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b",
    re.IGNORECASE | re.UNICODE | re.DOTALL,
)
_DATE_PATTERN = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s*(?P<month>{_MONTH_PATTERN})\b",
    re.IGNORECASE | re.UNICODE,
)
_LOCAL_CONTEXT_WORDS = re.compile(
    r"\b(волгоград|москва|санкт-петербург|амфитеатр|ресторан|храм|набережная)\b",
    re.IGNORECASE | re.UNICODE,
)
_DATETIME_OFFSET_SUFFIX = re.compile(r"([T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(?:Z|[+-]\d{2}:?\d{2})$")
_GIVEAWAY_RESULT_WORDS = re.compile(
    r"\b(итоги\s+(?:розыгрыша|конкурса)|поздравляем\s+победител|победител[ья]\b)",
    re.IGNORECASE | re.UNICODE,
)
_PAST_REPORT_WORDS = re.compile(
    r"\b(состоял(?:ся|ась|ось|ись)|прош[её]л|прошла|прошло|отметили|смотрели|собрал[аои]?|посетител(?:ями|и)\s+стал[ио]|"
    r"стал[ао]?\s+(?:для\s+\S+\s+)?(?:праздником|опытом|традицией)|спасибо\s+(?:каждому|всем))\b",
    re.IGNORECASE | re.UNICODE,
)
_ADMISSION_AD_WORDS = re.compile(
    r"\b(подать\s+документы|поступлени[ея]|прием\s+документов|приём\s+документов|"
    r"программа\s+(?:высшего\s+)?образования|колледж|университет)\b",
    re.IGNORECASE | re.UNICODE,
)
_APPLICATION_ACTIVITY_WORDS = re.compile(
    r"\b(при[её]м\s+заявок|подать\s+заявк[уыи]?|пода(?:й|вайте)\s+заявк[уыи]?|регистрац|дедлайн|голосовани[ея]\s+.+продолжается)\b",
    re.IGNORECASE | re.UNICODE,
)
_DEADLINE_DATE_PATTERN = re.compile(
    rf"\b(?:до|дедлайн:?)\s+(?P<day>\d{{1,2}})\s*(?P<month>{_MONTH_PATTERN})\b",
    re.IGNORECASE | re.UNICODE,
)
_EXPLICIT_END_TIME_WORDS = re.compile(
    r"\b(?:до|по|окончание|завершение|финал|итоги|результаты)\b",
    re.IGNORECASE | re.UNICODE,
)
_DUPLICATE_TITLE_STOPWORDS = {
    "в",
    "на",
    "и",
    "для",
    "по",
    "с",
    "со",
    "о",
    "об",
    "мероприятие",
    "событие",
    "фестиваль",
    "концерт",
    "спектакль",
    "конкурс",
    "кинопоказ",
    "вечер",
}


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
                "User-Agent": "event-extraction-agent/0.3",
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
        refinement_llm_client: LLMClient | None = None,
        config: ExtractionAgentConfig | None = None,
        current_datetime: str | None = None,
    ) -> None:
        self.config = config or ExtractionAgentConfig()
        self.llm_client = llm_client or self.config.main_client
        if self.llm_client is None:
            raise ValueError("llm_client is required; pass it directly or set config.main_client")
        self.refinement_llm_client = refinement_llm_client or self.config.refinement_client or self.llm_client
        self.current_datetime = current_datetime or self.config.current_datetime
        self.rate_limiter = (
            RequestRateLimiter(self.config.min_request_interval_seconds)
            if self.config.min_request_interval_seconds > 0
            else None
        )
        self.last_metadata: dict[str, Any] = {}

    def extract(self, post: SourcePost) -> ExtractionOutcome:
        return self._extract_once(post, client=self.llm_client, stage="main_extraction")

    def _extract_once(
        self,
        post: SourcePost,
        client: LLMClient,
        stage: str,
        previous_metadata: dict[str, Any] | None = None,
    ) -> ExtractionOutcome:
        current_datetime = self.current_datetime or _current_datetime()
        raw_text = post.raw_text_for_prompt()
        prompt = build_extraction_prompt(
            raw_text=raw_text,
            source_name=post.source_name,
            source_url=post.source_url,
            published_at=post.published_at_for_prompt(),
            external_id=post.external_id,
            current_datetime=current_datetime,
        )
        self.last_metadata = _metadata(
            client=client,
            main_client=self.llm_client,
            stage=stage,
            external_id=post.external_id,
            current_datetime=current_datetime,
            config=self.config,
            refinement_client=self.refinement_llm_client,
            previous_metadata=previous_metadata,
        )

        try:
            llm_content = self._complete(client, SYSTEM_PROMPT, prompt)
            response_payload = _parse_llm_json(llm_content)
        except Exception as exc:
            self._add_llm_attempt(stage=stage, client=client, success=False, error=str(exc))
            return _outcome(
                status=ExtractionStatus.LLM_ERROR,
                post=post,
                errors=[_error("llm", "llm_error", str(exc))],
                metadata=self.last_metadata,
        )
        self._add_llm_attempt(stage=stage, client=client, success=True)

        skip_reason = _obvious_non_announcement_reason(raw_text)
        if skip_reason is not None:
            return _outcome(
                status=ExtractionStatus.SKIPPED,
                post=post,
                errors=[_error("root", skip_reason, "post is not an event announcement")],
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

        published_at = post.published_at_for_prompt()
        event_payloads = _expand_event_payloads(
            _response_event_payloads(response_payload),
            raw_text=raw_text,
            published_at=published_at,
        )
        if not event_payloads:
            return _outcome(
                status=ExtractionStatus.LLM_ERROR,
                post=post,
                errors=[_error("event", "invalid_llm_shape", "LLM response must contain event object")],
                metadata=self.last_metadata,
            )

        validated_events: list[Event] = []
        validation_errors: list[ExtractionError] = []
        prefix_errors = len(event_payloads) > 1
        for event_index, event_payload in enumerate(event_payloads):
            event_payload = _repair_event_payload(
                event_payload,
                raw_text=raw_text,
                published_at=published_at,
            )
            event_payload = self._refine_event_type(event_payload, raw_text=raw_text)
            event_payload = _with_source_metadata(
                event_payload,
                raw_text=raw_text,
                source_name=post.source_name,
                source_url=post.source_url,
            )
            validation = validate_extraction_result(event_payload)
            if validation.is_valid and validation.event is not None:
                validated_events.append(validation.event)
            else:
                prefix = f"events.{event_index}" if prefix_errors else None
                validation_errors.extend(_issue_to_error(issue, prefix=prefix) for issue in validation.errors)

        if validation_errors:
            return _outcome(
                status=ExtractionStatus.INVALID,
                post=post,
                errors=validation_errors,
                metadata=self.last_metadata,
            )

        event = validated_events[0]
        return _outcome(
            status=ExtractionStatus.EXTRACTED,
            post=post,
            event=event,
            events=validated_events,
            metadata=self.last_metadata,
        )

    def _complete(self, client: LLMClient, system_prompt: str, user_prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            try:
                return client.complete(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise
        assert last_error is not None
        raise last_error

    def _add_llm_attempt(self, stage: str, client: LLMClient, success: bool, error: str | None = None) -> None:
        attempt: dict[str, Any] = {
            "stage": stage,
            "model": getattr(client, "model", None),
            "success": success,
        }
        if error is not None:
            attempt["error"] = error
        attempts = list(self.last_metadata.get("llm_attempts", []))
        attempts.append(attempt)
        self.last_metadata["llm_attempts"] = attempts

    def extract_many(
        self,
        posts: list[SourcePost],
        settings: BatchExtractionSettings | None = None,
    ) -> list[ExtractionOutcome]:
        return self.extract_batch(posts, settings=settings).outcomes

    def extract_batch(
        self,
        posts: list[SourcePost],
        settings: BatchExtractionSettings | None = None,
    ) -> BatchExtractionResult:
        batch_settings = settings or BatchExtractionSettings()
        outcomes: list[ExtractionOutcome] = []
        seen_keys: dict[str, int] = {}
        error_count = 0
        error_limit_reached = False

        for index, post in enumerate(posts):
            if _is_blank_text(post.raw_text_for_prompt()) and batch_settings.skip_empty:
                outcomes.append(
                    _outcome(
                        status=ExtractionStatus.SKIPPED,
                        post=post,
                        errors=[_error("post.text", "empty_post", "post text is empty")],
                        metadata={"batch_index": index},
                    )
                )
                continue

            duplicate_key = _post_deduplication_key(post)
            duplicate_of = seen_keys.get(duplicate_key)
            if duplicate_of is not None and batch_settings.skip_duplicates:
                outcomes.append(
                    _outcome(
                        status=ExtractionStatus.SKIPPED,
                        post=post,
                        errors=[_error("post", "duplicate_post", "post duplicates an earlier batch item")],
                        metadata={"batch_index": index, "duplicate_of": duplicate_of},
                    )
                )
                continue
            seen_keys[duplicate_key] = index

            if batch_settings.max_errors is not None and error_count >= batch_settings.max_errors:
                error_limit_reached = True
                outcomes.append(
                    _outcome(
                        status=ExtractionStatus.SKIPPED,
                        post=post,
                        errors=[_error("batch", "error_limit_reached", "batch error limit has been reached")],
                        metadata={"batch_index": index},
                    )
                )
                continue

            outcome = self.extract(post)
            metadata = dict(outcome.raw_llm_metadata or {})
            metadata["batch_index"] = index
            outcome = outcome.model_copy(update={"raw_llm_metadata": metadata})
            outcomes.append(outcome)

            if outcome.status in {ExtractionStatus.INVALID, ExtractionStatus.LLM_ERROR}:
                error_count += 1

        return BatchExtractionResult.from_outcomes(
            _deduplicate_event_outcomes(outcomes, batch_settings),
            settings=batch_settings,
            error_limit_reached=error_limit_reached,
        )

    def extract_incremental(
        self,
        posts: list[SourcePost],
        existing_outcomes: list[ExtractionOutcome],
        settings: BatchExtractionSettings | None = None,
        retry_llm_errors: bool = True,
    ) -> BatchExtractionResult:
        existing_index = _existing_outcomes_index(existing_outcomes)
        pending_posts: list[SourcePost] = []
        pending_positions: list[int] = []
        outcomes: list[ExtractionOutcome | None] = [None] * len(posts)

        for index, post in enumerate(posts):
            existing_outcome = _matching_existing_outcome(post, existing_index)
            if existing_outcome is None:
                pending_positions.append(index)
                pending_posts.append(post)
                continue

            if retry_llm_errors and existing_outcome.status == ExtractionStatus.LLM_ERROR:
                pending_positions.append(index)
                pending_posts.append(post)
                continue

            outcomes[index] = _cached_outcome(existing_outcome, post, source_index=index)

        if pending_posts:
            processed_result = self.extract_batch(pending_posts, settings=settings)
            for pending_index, outcome in enumerate(processed_result.outcomes):
                source_index = pending_positions[pending_index]
                metadata = dict(outcome.raw_llm_metadata or {})
                metadata["incremental_cached"] = False
                metadata["source_index"] = source_index
                outcomes[source_index] = outcome.model_copy(update={"raw_llm_metadata": metadata})

        return BatchExtractionResult.from_outcomes(
            _deduplicate_event_outcomes([outcome for outcome in outcomes if outcome is not None], settings),
            settings=settings,
        )

    def extract_event(self, post: SourcePost) -> Event:
        outcome = self.extract(post)
        if outcome.status != ExtractionStatus.EXTRACTED or outcome.event is None:
            raise ValueError(f"event extraction failed: {outcome.status}: {outcome.errors}")
        return outcome.event

    def _refine_event_type(self, event_payload: dict[str, Any], raw_text: str) -> dict[str, Any]:
        copied = dict(event_payload)
        if not self.config.use_event_type_refinement:
            self.last_metadata["event_type_refinement"] = "disabled"
            return copied

        event_type = copied.get("event_type")
        if event_type in EVENT_TYPE_VALUES and event_type != "SocialEvent":
            self.last_metadata["event_type_refinement"] = "not_needed"
            return copied

        prompt = build_event_type_classification_prompt(raw_text=raw_text, draft=copied)
        try:
            content = self._complete(self.refinement_llm_client, EVENT_TYPE_CLASSIFICATION_PROMPT, prompt)
            payload = _parse_llm_json(content)
        except Exception as exc:
            self._add_llm_attempt(
                stage="event_type_refinement",
                client=self.refinement_llm_client,
                success=False,
                error=str(exc),
            )
            self.last_metadata["event_type_refine_error"] = str(exc)
            if event_type not in EVENT_TYPE_VALUES:
                copied["event_type"] = "SocialEvent"
            return copied
        self._add_llm_attempt(stage="event_type_refinement", client=self.refinement_llm_client, success=True)

        refined_event_type = payload.get("event_type")
        if refined_event_type in EVENT_TYPE_VALUES:
            copied["event_type"] = refined_event_type
            self.last_metadata["event_type_refined"] = True
            self.last_metadata["event_type_refinement"] = "completed"
        elif event_type not in EVENT_TYPE_VALUES:
            copied["event_type"] = "SocialEvent"
            self.last_metadata["event_type_refinement"] = "defaulted"
        return copied


def _parse_llm_json(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON response must be an object")
    return payload


def _response_event_payloads(response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = response_payload.get("events")
    if isinstance(events, list):
        payloads = [item for item in events if isinstance(item, dict)]
        if payloads:
            return payloads
    event = response_payload.get("event")
    if isinstance(event, dict):
        return [event]
    return []


def _expand_event_payloads(payloads: list[dict[str, Any]], raw_text: str, published_at: str | None) -> list[dict[str, Any]]:
    if len(payloads) != 1:
        return payloads
    repeated_starts = _non_contiguous_repeated_starts(raw_text, published_at)
    if len(repeated_starts) < 2:
        return payloads
    expanded = []
    for start_at in repeated_starts:
        copied = dict(payloads[0])
        copied["start_at"] = start_at
        copied["end_at"] = None
        expanded.append(copied)
    return expanded


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
    if _is_missing_value(copied.get("timezone")):
        copied["timezone"] = "Europe/Moscow" if _LOCAL_CONTEXT_WORDS.search(raw_text) else "unknown"
    _strip_datetime_offsets(copied)
    if _is_missing_value(copied.get("start_at")):
        inferred_start_at = _extract_start_at(raw_text, published_at)
        if inferred_start_at is not None:
            copied["start_at"] = inferred_start_at
    _repair_application_dates(copied, raw_text=raw_text, published_at=published_at)
    _drop_inferred_duration_end_at(copied, raw_text=raw_text)
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


def _repair_application_dates(payload: dict[str, Any], raw_text: str, published_at: str | None) -> None:
    if not _APPLICATION_ACTIVITY_WORDS.search(raw_text):
        return
    deadline = _extract_deadline_at(raw_text, published_at)
    published_date = _published_date(published_at)
    if deadline is None or published_date is None:
        return

    if _same_date(payload.get("start_at"), deadline):
        payload["start_at"] = datetime.combine(published_date, time()).isoformat()
        if _is_missing_value(payload.get("end_at")):
            payload["end_at"] = deadline


def _drop_inferred_duration_end_at(payload: dict[str, Any], raw_text: str) -> None:
    start_at = _parse_datetime_value(payload.get("start_at"))
    end_at = _parse_datetime_value(payload.get("end_at"))
    if start_at is None or end_at is None:
        return
    if start_at.date() != end_at.date() or end_at <= start_at:
        return
    if _EXPLICIT_END_TIME_WORDS.search(raw_text):
        return
    if "продолжительность" in raw_text.lower() or re.search(r"\b\d+(?:[,.]\d+)?\s*час", raw_text, re.IGNORECASE):
        payload["end_at"] = None


def _obvious_non_announcement_reason(raw_text: str) -> str | None:
    if _GIVEAWAY_RESULT_WORDS.search(raw_text):
        return "not_event_announcement"
    if _PAST_REPORT_WORDS.search(raw_text):
        return "past_event_report"
    if _ADMISSION_AD_WORDS.search(raw_text) and _extract_start_at(raw_text, None) is None:
        return "not_event_announcement"
    return None


def _strip_datetime_offsets(payload: dict[str, Any]) -> None:
    for field in ("start_at", "end_at"):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = _DATETIME_OFFSET_SUFFIX.sub(r"\1", value)


def _extract_start_at(raw_text: str, published_at: str | None) -> str | None:
    match = _DATE_RANGE_TIME_PATTERN.search(raw_text)
    if match is not None:
        return _build_start_at(
            day=int(match.group("day")),
            month=_MONTHS[match.group("month").lower()],
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
            published_at=published_at,
        )

    match = _DATE_TIME_PATTERN.search(raw_text)
    if match is not None:
        return _build_start_at(
            day=int(match.group("day")),
            month=_MONTHS[match.group("month").lower()],
            hour=int(match.group("hour")),
            minute=int(match.group("minute")),
            published_at=published_at,
        )

    match = _DATE_PATTERN.search(raw_text)
    if match is None:
        return None

    return _build_start_at(
        day=int(match.group("day")),
        month=_MONTHS[match.group("month").lower()],
        hour=0,
        minute=0,
        published_at=published_at,
    )


def _extract_deadline_at(raw_text: str, published_at: str | None) -> str | None:
    match = _DEADLINE_DATE_PATTERN.search(raw_text)
    if match is None:
        return None
    return _build_start_at(
        day=int(match.group("day")),
        month=_MONTHS[match.group("month").lower()],
        hour=0,
        minute=0,
        published_at=published_at,
    )


def _non_contiguous_repeated_starts(raw_text: str, published_at: str | None) -> list[str]:
    match = _LISTED_DATES_TIME_PATTERN.search(raw_text)
    if match is None:
        return []
    days = [int(value) for value in re.findall(r"\d{1,2}", match.group("days"))]
    if len(days) < 2 or _is_consecutive(days):
        return []

    month = _MONTHS[match.group("month").lower()]
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    starts = [
        _build_start_at(day=day, month=month, hour=hour, minute=minute, published_at=published_at)
        for day in days
    ]
    return [value for value in starts if value is not None]


def _is_consecutive(values: list[int]) -> bool:
    ordered = sorted(values)
    return all(right - left == 1 for left, right in zip(ordered, ordered[1:]))


def _build_start_at(day: int, month: int, hour: int, minute: int, published_at: str | None) -> str | None:
    if not (1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    event_date = _resolve_event_date(day, month, published_at)
    if event_date is None:
        return None

    event_datetime = datetime.combine(event_date, time(hour, minute))
    return event_datetime.isoformat()


def _resolve_event_date(day: int, month: int, published_at: str | None) -> date | None:
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

    return event_date


def _deduplicate_event_outcomes(
    outcomes: list[ExtractionOutcome],
    settings: BatchExtractionSettings | None,
) -> list[ExtractionOutcome]:
    if settings is not None and not settings.skip_event_duplicates:
        return outcomes

    extracted_indices = [
        index
        for index, outcome in enumerate(outcomes)
        if outcome.status == ExtractionStatus.EXTRACTED and outcome.event is not None
    ]
    groups: list[list[int]] = []

    # ponytail: O(n²) is fine for batch post-processing; index later if batch sizes hurt.
    for index in extracted_indices:
        group = next(
            (
                candidate
                for candidate in groups
                if any(_events_are_duplicates(outcomes[index], outcomes[other]) for other in candidate)
            ),
            None,
        )
        if group is None:
            groups.append([index])
        else:
            group.append(index)

    deduplicated = list(outcomes)
    for group in groups:
        if len(group) < 2:
            continue
        keep_index = max(group, key=lambda index: (_published_at_sort_key(outcomes[index].post), index))
        for index in group:
            if index != keep_index:
                deduplicated[index] = _duplicate_event_outcome(outcomes[index], keep_index)
    return deduplicated


def _events_are_duplicates(left: ExtractionOutcome, right: ExtractionOutcome) -> bool:
    if left.event is None or right.event is None:
        return False
    title_score = _title_containment(left.event.title, right.event.title)
    if title_score < 0.65:
        return False
    if not _event_dates_are_close(left.event.start_at, right.event.start_at) and not _texts_share_event_date(
        left.post.raw_text_for_prompt(),
        right.post.raw_text_for_prompt(),
        title_score=title_score,
    ):
        return False
    return _locations_are_compatible(left.event.city, right.event.city) and _locations_are_compatible(
        left.event.venue_name,
        right.event.venue_name,
    )


def _duplicate_event_outcome(outcome: ExtractionOutcome, keep_index: int) -> ExtractionOutcome:
    metadata = dict(outcome.raw_llm_metadata or {})
    metadata["duplicate_of"] = keep_index
    return outcome.model_copy(
        update={
            "status": ExtractionStatus.SKIPPED,
            "event": None,
            "events": None,
            "errors": [_error("event", "duplicate_event", "event duplicates a later extracted event")],
            "raw_llm_metadata": metadata,
        }
    )


def _title_containment(left: str, right: str) -> float:
    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _title_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-zа-яё0-9]+", value.lower().replace("ё", "е"), re.IGNORECASE))
    significant = tokens - _DUPLICATE_TITLE_STOPWORDS
    return significant or tokens


def _event_dates_are_close(left: datetime, right: datetime) -> bool:
    return abs((left.date() - right.date()).days) <= 3


def _texts_share_event_date(left: str, right: str, *, title_score: float) -> bool:
    if title_score < 0.8:
        return False
    return bool(_text_date_tokens(left) & _text_date_tokens(right))


def _text_date_tokens(value: str) -> set[str]:
    return {f"{match.group('day')}.{_MONTHS[match.group('month').lower()]}" for match in _DATE_PATTERN.finditer(value)}


def _same_date(value: Any, iso_datetime: str) -> bool:
    left = _parse_datetime_value(value)
    right = _parse_datetime_value(iso_datetime)
    return left is not None and right is not None and left.date() == right.date()


def _parse_datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(_DATETIME_OFFSET_SUFFIX.sub(r"\1", value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _locations_are_compatible(left: str | None, right: str | None) -> bool:
    left_tokens = _location_tokens(left)
    right_tokens = _location_tokens(right)
    if not left_tokens or not right_tokens:
        return True
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)) >= 0.5


def _location_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return set(re.findall(r"[a-zа-яё0-9]+", value.lower().replace("ё", "е"), re.IGNORECASE))


def _published_at_sort_key(post: SourcePost) -> float:
    value = post.published_at
    if value is None:
        return float("-inf")
    if isinstance(value, datetime):
        published_at = value
    else:
        try:
            published_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return float("-inf")
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    return published_at.timestamp()


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


def _is_blank_text(value: str) -> bool:
    return not value.strip()


def _post_deduplication_key(post: SourcePost) -> str:
    if post.external_id:
        return f"external_id:{post.external_id}"
    return "text:" + " ".join(post.raw_text_for_prompt().casefold().split())


def _existing_outcomes_index(outcomes: list[ExtractionOutcome]) -> dict[str, ExtractionOutcome]:
    index: dict[str, ExtractionOutcome] = {}
    for outcome in outcomes:
        external_id = outcome.post.external_id
        if external_id is not None:
            index[external_id] = outcome
    return index


def _matching_existing_outcome(
    post: SourcePost,
    existing_index: dict[str, ExtractionOutcome],
) -> ExtractionOutcome | None:
    if post.external_id is None:
        return None

    existing_outcome = existing_index.get(post.external_id)
    if existing_outcome is None:
        return None

    if _normalized_incremental_text(existing_outcome.post.raw_text_for_prompt()) != _normalized_incremental_text(
        post.raw_text_for_prompt()
    ):
        return None

    return existing_outcome


def _cached_outcome(existing_outcome: ExtractionOutcome, post: SourcePost, source_index: int) -> ExtractionOutcome:
    metadata = dict(existing_outcome.raw_llm_metadata or {})
    metadata["incremental_cached"] = True
    metadata["source_index"] = source_index
    return existing_outcome.model_copy(update={"post": post, "raw_llm_metadata": metadata})


def _normalized_incremental_text(value: str) -> str:
    return " ".join(value.split())


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


def _metadata(
    client: LLMClient,
    main_client: LLMClient,
    stage: str,
    external_id: str | None,
    current_datetime: str,
    config: ExtractionAgentConfig,
    refinement_client: LLMClient | None,
    previous_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_model = _client_model_name(client, main_client, refinement_client, config)
    metadata = {
        "llm_model": active_model,
        "main_model": getattr(main_client, "model", None) or config.main_model,
        "refinement_model": getattr(refinement_client, "model", None) or config.refinement_model,
        "active_model": active_model,
        "active_stage": stage,
        "external_id": external_id,
        "current_datetime": current_datetime,
        "event_type_refinement_enabled": config.use_event_type_refinement,
        "request_timeout_seconds": config.request_timeout_seconds,
        "min_request_interval_seconds": config.min_request_interval_seconds,
        "max_retries": config.max_retries,
    }
    if previous_metadata is not None:
        metadata["previous_llm_metadata"] = previous_metadata
        metadata["llm_attempts"] = list(previous_metadata.get("llm_attempts", []))
    return metadata


def _client_model_name(
    client: LLMClient,
    main_client: LLMClient,
    refinement_client: LLMClient | None,
    config: ExtractionAgentConfig,
) -> str | None:
    model_name = getattr(client, "model", None)
    if model_name is not None:
        return model_name
    if client is main_client:
        return config.main_model
    if refinement_client is not None and client is refinement_client:
        return config.refinement_model
    return None


def _outcome(
    status: ExtractionStatus,
    post: SourcePost,
    event: Event | None = None,
    events: list[Event] | None = None,
    errors: list[ExtractionError] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExtractionOutcome:
    return ExtractionOutcome(
        status=status,
        event=event,
        events=events,
        post=post,
        errors=errors or [],
        raw_llm_metadata=metadata or None,
    )


def _issue_to_error(issue: ValidationIssue, prefix: str | None = None) -> ExtractionError:
    field = f"{prefix}.{issue.field}" if prefix else issue.field
    return _error(field, issue.code, issue.message)


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
