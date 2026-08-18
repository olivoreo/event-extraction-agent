from __future__ import annotations

import json
import re
import threading
import time as time_module
import urllib.error
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

from event_extraction_agent.models import (
    BatchExtractionResult,
    BatchExtractionSettings,
    DuplicateExtractedEvent,
    Event,
    ExtractedEvent,
    ExtractionAgentConfig,
    ExtractionError,
    ExtractionOutcome,
    ExtractionStatus,
    SourcePost,
)
from event_extraction_agent.prompts import (
    ATTENDANCE_TYPE_VALUES,
    EVENT_TYPE_CLASSIFICATION_PROMPT,
    EVENT_TYPE_VALUES,
    INDUSTRY_VALUES,
    ROLE_VALUES,
    SKIP_REASONS,
    SYSTEM_PROMPT,
    TITLE_DESCRIPTION_REFINEMENT_PROMPT,
    build_event_type_classification_prompt,
    build_extraction_prompt,
    build_invalid_date_repair_prompt,
    build_title_description_refinement_prompt,
    event_type_json_schema,
    extraction_json_schema,
    title_description_json_schema,
)
from event_extraction_agent.validator import ValidationIssue, validate_extraction_result

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MAIN_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_GROQ_MAX_RETRIES = 3
GROQ_RATE_LIMIT_BUFFER_SECONDS = 0.25
GROQ_RATE_LIMIT_FALLBACK_SECONDS = 60.0
GROQ_STRICT_SCHEMA_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
GROQ_SCHEMA_MODELS = {
    *GROQ_STRICT_SCHEMA_MODELS,
    "openai/gpt-oss-safeguard-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

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
_DATE_RANGE_PATTERN = re.compile(
    rf"\b(?P<start_day>\d{{1,2}})\s*(?P<connector>и|[-–—])\s*"
    rf"(?P<end_day>\d{{1,2}})\s*(?P<month>{_MONTH_PATTERN})\b",
    re.IGNORECASE | re.UNICODE,
)
_DATE_FROM_TO_RANGE_PATTERN = re.compile(
    rf"\bс\s+(?P<start_day>\d{{1,2}})(?:\s*(?P<start_month>{_MONTH_PATTERN}))?\s+по\s+"
    rf"(?P<end_day>\d{{1,2}})\s*(?P<end_month>{_MONTH_PATTERN})\b",
    re.IGNORECASE | re.UNICODE,
)
_DATE_CROSS_MONTH_RANGE_PATTERN = re.compile(
    rf"\b(?P<start_day>\d{{1,2}})\s*(?P<start_month>{_MONTH_PATTERN})\s*[-–—]\s*"
    rf"(?P<end_day>\d{{1,2}})\s*(?P<end_month>{_MONTH_PATTERN})\b",
    re.IGNORECASE | re.UNICODE,
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
_CANCELLATION_UPDATE_WORDS = re.compile(
    r"\b(отмена|отмен[её]н[аоы]?|отменяется|отменили|не\s+состоится)\b",
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


class GroqDailyRateLimitError(RuntimeError):
    """Raised when Groq's requests-per-day or tokens-per-day quota is exhausted."""


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
        return self._complete(system_prompt, user_prompt, {"type": "json_object"})

    def complete_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        if self.model not in GROQ_SCHEMA_MODELS:
            return self.complete(system_prompt, user_prompt)
        return self._complete(
            system_prompt,
            user_prompt,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "event_extraction",
                    "strict": self.model in GROQ_STRICT_SCHEMA_MODELS,
                    "schema": response_schema,
                },
            },
        )

    def _complete(self, system_prompt: str, user_prompt: str, response_format: dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "temperature": 0,
            "response_format": response_format,
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
        retries = 0
        while True:
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace").strip()
                if exc.code == 429:
                    if _is_daily_groq_rate_limit(exc, error_body):
                        raise GroqDailyRateLimitError(_groq_http_error_detail(exc, error_body)) from exc
                    time_module.sleep(_groq_minute_rate_limit_delay(exc, error_body))
                    continue
                if 500 <= exc.code <= 599 and retries < self.max_retries:
                    _sleep_before_retry(exc, retries)
                    retries += 1
                    continue
                raise RuntimeError(f"Groq request failed:{_groq_http_error_detail(exc, error_body)}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Groq request failed: {exc}") from exc


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
        self.title_description_refinement_llm_client = refinement_llm_client or self.config.refinement_client or self.llm_client
        if self.config.use_event_type_refinement:
            self.refinement_llm_client = refinement_llm_client or self.config.refinement_client or self.llm_client
        else:
            self.refinement_llm_client = None
        self.current_datetime = current_datetime or self.config.current_datetime
        self.rate_limiter = (
            RequestRateLimiter(self.config.min_request_interval_seconds)
            if self.config.min_request_interval_seconds > 0
            else None
        )
        self.last_metadata: dict[str, Any] = {}

    def extract(self, post: SourcePost) -> ExtractionOutcome:
        try:
            return self._extract_once(post, client=self.llm_client, stage="main_extraction")
        except GroqDailyRateLimitError as exc:
            return _outcome(
                status=ExtractionStatus.LLM_ERROR,
                post=post,
                errors=[_error("llm", "daily_rate_limit_exceeded", str(exc))],
                metadata=self.last_metadata,
            )

    def _extract_once(
        self,
        post: SourcePost,
        client: LLMClient,
        stage: str,
        previous_metadata: dict[str, Any] | None = None,
        repair_errors: list[ExtractionError] | None = None,
    ) -> ExtractionOutcome:
        current_datetime = self.current_datetime or _current_datetime()
        raw_text = post.raw_text_for_prompt()
        skip_reason = _obvious_non_announcement_reason(raw_text)
        if skip_reason is not None:
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
            return _outcome(
                status=ExtractionStatus.SKIPPED,
                post=post,
                errors=[_error("root", skip_reason, "post is not an event announcement")],
                metadata=self.last_metadata,
            )
        prompt_kwargs = {
            "raw_text": raw_text,
            "source_name": post.source_name,
            "source_url": post.source_url,
            "published_at": post.published_at_for_prompt(),
            "external_id": post.external_id,
            "current_datetime": current_datetime,
        }
        prompt = (
            build_extraction_prompt(**prompt_kwargs)
            if repair_errors is None
            else build_invalid_date_repair_prompt(
                **prompt_kwargs,
                previous_errors=[error.model_dump(mode="json") for error in repair_errors],
            )
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
            llm_content = self._complete(client, SYSTEM_PROMPT, prompt, extraction_json_schema())
            response_payload = _parse_llm_json(llm_content)
        except Exception as exc:
            self._add_llm_attempt(stage=stage, client=client, success=False, error=str(exc))
            return _outcome(
                status=ExtractionStatus.LLM_ERROR,
                post=post,
                errors=[
                    _error(
                        "llm",
                        "daily_rate_limit_exceeded" if isinstance(exc, GroqDailyRateLimitError) else "llm_error",
                        str(exc),
                    )
                ],
                metadata=self.last_metadata,
        )
        self._add_llm_attempt(stage=stage, client=client, success=True)

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
            event_payload = self._refine_title_description(event_payload, raw_text=raw_text)
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
            if repair_errors is None and _should_retry_invalid_date_order(validation_errors):
                return self._extract_once(
                    post,
                    client=self.llm_client,
                    stage="invalid_date_repair",
                    previous_metadata=self.last_metadata,
                    repair_errors=validation_errors,
                )
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
            events=validated_events if len(validated_events) > 1 else None,
            metadata=self.last_metadata,
        )

    def _complete(
        self,
        client: LLMClient,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            try:
                complete_with_schema = getattr(client, "complete_with_schema", None)
                if response_schema is not None and callable(complete_with_schema):
                    return complete_with_schema(system_prompt, user_prompt, response_schema)
                return client.complete(system_prompt, user_prompt)
            except Exception as exc:
                last_error = exc
                if isinstance(exc, GroqDailyRateLimitError):
                    raise
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
        daily_limit_error: ExtractionError | None = None

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

            if daily_limit_error is not None:
                outcomes.append(
                    _outcome(
                        status=ExtractionStatus.LLM_ERROR,
                        post=post,
                        errors=[daily_limit_error],
                        metadata={"batch_index": index},
                    )
                )
                error_count += 1
                continue

            outcome = self.extract(post)
            metadata = dict(outcome.raw_llm_metadata or {})
            metadata["batch_index"] = index
            outcome = outcome.model_copy(update={"raw_llm_metadata": metadata})
            outcomes.append(outcome)

            if outcome.status in {ExtractionStatus.INVALID, ExtractionStatus.LLM_ERROR}:
                error_count += 1
            daily_limit_error = next(
                (error for error in outcome.errors if error.code == "daily_rate_limit_exceeded"),
                daily_limit_error,
            )

        return _batch_result(outcomes, settings=batch_settings, error_limit_reached=error_limit_reached)

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

        return _batch_result([outcome for outcome in outcomes if outcome is not None], settings=settings)

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
            content = self._complete(
                self.refinement_llm_client,
                EVENT_TYPE_CLASSIFICATION_PROMPT,
                prompt,
                event_type_json_schema(),
            )
            payload = _parse_llm_json(content)
        except Exception as exc:
            self._add_llm_attempt(
                stage="event_type_refinement",
                client=self.refinement_llm_client,
                success=False,
                error=str(exc),
            )
            self.last_metadata["event_type_refine_error"] = str(exc)
            if isinstance(exc, GroqDailyRateLimitError):
                raise
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

    def _refine_title_description(self, event_payload: dict[str, Any], raw_text: str) -> dict[str, Any]:
        copied = dict(event_payload)
        if not self.config.use_title_description_refinement:
            self.last_metadata["title_description_refinement"] = "disabled"
            return copied
        if self.title_description_refinement_llm_client is None:
            self.last_metadata["title_description_refinement"] = "unavailable"
            return copied
        client = self.title_description_refinement_llm_client
        prompt = build_title_description_refinement_prompt(raw_text=raw_text, draft=copied)
        try:
            content = self._complete(
                client,
                TITLE_DESCRIPTION_REFINEMENT_PROMPT,
                prompt,
                title_description_json_schema(),
            )
            payload = _parse_llm_json(content)
        except Exception as exc:
            self._add_llm_attempt(
                stage="title_description_refinement",
                client=client,
                success=False,
                error=str(exc),
            )
            self.last_metadata["title_description_refine_error"] = str(exc)
            self.last_metadata["title_description_refinement"] = "failed"
            if isinstance(exc, GroqDailyRateLimitError):
                raise
            return copied
        self._add_llm_attempt(stage="title_description_refinement", client=client, success=True)

        title = payload.get("title")
        if isinstance(title, str) and title.strip() and _title_matches_raw_text(title, raw_text):
            copied["title"] = title.strip()
            self.last_metadata["title_description_refined"] = True
            self.last_metadata["title_description_refinement"] = "completed"
        else:
            self.last_metadata["title_description_refinement"] = "rejected"

        description = payload.get("description")
        if description is None or isinstance(description, str):
            copied["description"] = description.strip() if isinstance(description, str) else None
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


def _repair_event_payload(
    payload: dict[str, Any],
    raw_text: str,
    published_at: str | None,
) -> dict[str, Any]:
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
    _repair_explicit_date_range_end_at(copied, raw_text=raw_text)
    _coerce_prompt_enum(copied, "attendance_type", ATTENDANCE_TYPE_VALUES)
    _filter_prompt_list(copied, "relevant_roles", ROLE_VALUES)
    _filter_prompt_list(copied, "industries", INDUSTRY_VALUES)
    skills = copied.get("skills")
    if skills is not None:
        copied["skills"] = (
            [item.strip() for item in skills if isinstance(item, str) and item.strip()]
            if isinstance(skills, list)
            else None
        )
    if copied.get("attendance_type") == "unknown":
        copied["attendance_type"] = "OfflineEventAttendanceMode"
    return copied


def _repair_application_dates(payload: dict[str, Any], raw_text: str, published_at: str | None) -> None:
    if not _APPLICATION_ACTIVITY_WORDS.search(raw_text):
        return
    deadline = _extract_deadline_at(raw_text, published_at)
    if deadline is None:
        return

    if _same_date(payload.get("start_at"), deadline):
        payload["start_at"] = None
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


def _repair_explicit_date_range_end_at(payload: dict[str, Any], raw_text: str) -> None:
    if not _is_missing_value(payload.get("end_at")):
        return
    start_at = _parse_datetime_value(payload.get("start_at"))
    if start_at is None:
        return

    end_date = _explicit_range_end_date(raw_text, start_at.date())
    if end_date is None:
        return
    payload["end_at"] = datetime.combine(end_date, time.min).isoformat()


def _explicit_range_end_date(raw_text: str, start_date: date) -> date | None:
    for match in _DATE_CROSS_MONTH_RANGE_PATTERN.finditer(raw_text):
        candidate = _range_end_date_from_match(match, start_date)
        if candidate is not None:
            return candidate

    for match in _DATE_FROM_TO_RANGE_PATTERN.finditer(raw_text):
        candidate = _range_end_date_from_match(match, start_date)
        if candidate is not None:
            return candidate

    for match in _DATE_RANGE_PATTERN.finditer(raw_text):
        start_day = int(match.group("start_day"))
        end_day = int(match.group("end_day"))
        month = _MONTHS[match.group("month").lower()]
        if (start_date.day, start_date.month) != (start_day, month):
            continue
        if match.group("connector").lower() == "и" and end_day != start_day + 1:
            continue
        try:
            candidate = date(start_date.year, month, end_day)
        except ValueError:
            continue
        if candidate > start_date:
            return candidate
    return None


def _range_end_date_from_match(match: re.Match[str], start_date: date) -> date | None:
    start_day = int(match.group("start_day"))
    end_day = int(match.group("end_day"))
    start_month_name = match.groupdict().get("start_month") or match.groupdict().get("end_month")
    end_month_name = match.groupdict().get("end_month")
    if start_month_name is None or end_month_name is None:
        return None
    start_month = _MONTHS[start_month_name.lower()]
    end_month = _MONTHS[end_month_name.lower()]
    if (start_date.day, start_date.month) != (start_day, start_month):
        return None

    end_year = start_date.year + (1 if (end_month, end_day) < (start_month, start_day) else 0)
    try:
        candidate = date(end_year, end_month, end_day)
    except ValueError:
        return None
    return candidate if candidate > start_date else None


def _obvious_non_announcement_reason(raw_text: str) -> str | None:
    if _CANCELLATION_UPDATE_WORDS.search(raw_text):
        return "not_event_announcement"
    if _GIVEAWAY_RESULT_WORDS.search(raw_text):
        return "not_event_announcement"
    if _PAST_REPORT_WORDS.search(raw_text):
        return "past_event_report"
    if _ADMISSION_AD_WORDS.search(raw_text) and _extract_start_at(raw_text, None) is None:
        return "not_event_announcement"
    return None


def _strip_datetime_offsets(payload: dict[str, Any]) -> None:
    for field in ("start_at", "end_at"):
        parsed = _parse_datetime_value(payload.get(field))
        if parsed is not None:
            payload[field] = parsed.isoformat()


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


def _batch_result(
    outcomes: list[ExtractionOutcome],
    settings: BatchExtractionSettings | None,
    error_limit_reached: bool = False,
) -> BatchExtractionResult:
    events, duplicate_events = _deduplicate_extracted_events(
        BatchExtractionResult.from_outcomes(outcomes, settings=settings).events,
        settings,
    )
    return BatchExtractionResult.from_outcomes(
        outcomes,
        settings=settings,
        error_limit_reached=error_limit_reached,
        events=events,
        duplicate_events=duplicate_events,
    )


def _deduplicate_extracted_events(
    events: list[ExtractedEvent],
    settings: BatchExtractionSettings | None,
) -> tuple[list[ExtractedEvent], list[DuplicateExtractedEvent]]:
    if settings is not None and not settings.skip_event_duplicates:
        return events, []

    groups: list[list[int]] = []

    # ponytail: O(n²) is fine for batch post-processing; index later if batch sizes hurt.
    for index in range(len(events)):
        group = next(
            (
                candidate
                for candidate in groups
                if any(_events_are_duplicates(events[index], events[other]) for other in candidate)
            ),
            None,
        )
        if group is None:
            groups.append([index])
        else:
            group.append(index)

    duplicate_indices: set[int] = set()
    duplicate_events: list[DuplicateExtractedEvent] = []
    for group in groups:
        if len(group) < 2:
            continue
        keep_index = _duplicate_group_keep_index(group, events)
        for index in group:
            if index != keep_index:
                duplicate_indices.add(index)
                duplicate_events.append(
                    DuplicateExtractedEvent(
                        **events[index].model_dump(),
                        duplicate_of=keep_index,
                    )
                )
    return [event for index, event in enumerate(events) if index not in duplicate_indices], duplicate_events


def _duplicate_group_keep_index(group: list[int], events: list[ExtractedEvent]) -> int:
    with_dates = [index for index in group if _published_at_sort_key(events[index].post) != float("-inf")]
    if with_dates:
        return max(with_dates, key=lambda index: (_published_at_sort_key(events[index].post), index))
    return min(group)


def _events_are_duplicates(left: ExtractedEvent, right: ExtractedEvent) -> bool:
    title_score = _title_containment(left.event.title, right.event.title)
    description_score = _text_containment(left.event.description, right.event.description)
    if title_score < 0.65:
        return False
    dates_match = _event_dates_are_close(left.event.start_at, right.event.start_at)
    if not dates_match and (left.event.start_at is None or right.event.start_at is None):
        dates_match = _texts_share_event_date(
            left.post.raw_text_for_prompt(),
            right.post.raw_text_for_prompt(),
            title_score=title_score,
        )
    if not dates_match:
        return False
    if not _event_types_are_compatible(left.event.event_type, right.event.event_type) and not (
        title_score >= 0.8
        and left.event.start_at is not None
        and right.event.start_at is not None
        and left.event.start_at == right.event.start_at
    ):
        return False
    if not _industries_are_compatible(left.event.industries, right.event.industries):
        return False
    if title_score < 0.8 and left.event.description and right.event.description and description_score < 0.2:
        return False
    return True


def _title_containment(left: str, right: str) -> float:
    left_tokens = _title_tokens(left)
    right_tokens = _title_tokens(right)
    return _token_containment(left_tokens, right_tokens)


def _text_containment(left: str | None, right: str | None) -> float:
    left_tokens = _title_tokens(left or "")
    right_tokens = _title_tokens(right or "")
    return _token_containment(left_tokens, right_tokens)


def _token_containment(left_tokens: set[str], right_tokens: set[str]) -> float:
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _title_tokens(value: str) -> set[str]:
    tokens = set(re.findall(r"[a-zа-яё0-9]+", value.lower().replace("ё", "е"), re.IGNORECASE))
    significant = tokens - _DUPLICATE_TITLE_STOPWORDS
    return significant or tokens


def _title_matches_raw_text(title: Any, raw_text: str) -> bool:
    if not isinstance(title, str) or not title.strip():
        return False
    title_tokens = _title_tokens(title)
    text_tokens = _title_tokens(raw_text)
    if not title_tokens:
        return False
    return len(title_tokens & text_tokens) / len(title_tokens) >= 0.5


def _event_dates_are_close(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return False
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
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(_DATETIME_OFFSET_SUFFIX.sub(r"\1", value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def _event_types_are_compatible(left: Any, right: Any) -> bool:
    weak = {"unknown", "other"}
    left_value = getattr(left, "value", left)
    right_value = getattr(right, "value", right)
    return left_value in weak or right_value in weak or left_value == right_value


def _industries_are_compatible(left: list[str] | None, right: list[str] | None) -> bool:
    if not left or not right:
        return True
    return bool(set(left) & set(right))


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


def _groq_http_error_detail(exc: urllib.error.HTTPError, error_body: str) -> str:
    return f" HTTP {exc.code}: {error_body}" if error_body else f" HTTP {exc.code}"


def _is_daily_groq_rate_limit(exc: urllib.error.HTTPError, error_body: str) -> bool:
    if re.search(r"\b(?:RPD|TPD)\b|(?:requests|tokens)\s+per\s+day", error_body, re.IGNORECASE):
        return True
    remaining_requests = exc.headers.get("x-ratelimit-remaining-requests") if exc.headers is not None else None
    try:
        return remaining_requests is not None and float(remaining_requests) <= 0
    except ValueError:
        return False


def _groq_minute_rate_limit_delay(exc: urllib.error.HTTPError, error_body: str) -> float:
    headers = exc.headers
    retry_after = headers.get("Retry-After") if headers is not None else None
    delay = _retry_after_seconds(retry_after)
    if delay is None and headers is not None:
        delay = _parse_groq_duration(headers.get("x-ratelimit-reset-tokens"))
    if delay is None:
        match = re.search(
            r"try\s+again\s+in\s+((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h)\s*)+)",
            error_body,
            re.IGNORECASE,
        )
        delay = _parse_groq_duration(match.group(1)) if match else None
    return max(0.0, delay if delay is not None else GROQ_RATE_LIMIT_FALLBACK_SECONDS) + GROQ_RATE_LIMIT_BUFFER_SECONDS


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _parse_groq_duration(value: str | None) -> float | None:
    if not value:
        return None
    units = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", value, re.IGNORECASE)
    if not parts:
        return None
    return sum(float(amount) * units[unit.lower()] for amount, unit in parts)


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
    active_model = _client_model_name(client)
    metadata = {
        "llm_model": active_model,
        "main_model": getattr(main_client, "model", None),
        "refinement_model": getattr(refinement_client, "model", None),
        "active_model": active_model,
        "active_stage": stage,
        "external_id": external_id,
        "current_datetime": current_datetime,
        "event_type_refinement_enabled": config.use_event_type_refinement,
        "min_request_interval_seconds": config.min_request_interval_seconds,
        "max_retries": config.max_retries,
    }
    if previous_metadata is not None:
        metadata["previous_llm_metadata"] = previous_metadata
        metadata["llm_attempts"] = list(previous_metadata.get("llm_attempts", []))
    return metadata


def _client_model_name(
    client: LLMClient,
) -> str | None:
    return getattr(client, "model", None)


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


def _should_retry_invalid_date_order(errors: list[ExtractionError]) -> bool:
    return any(
        error.code == "invalid_field"
        and error.field.endswith("end_at")
        and error.message == "end_at must not be before start_at"
        for error in errors
    )


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
