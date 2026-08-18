from __future__ import annotations

import json
import re
import threading
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from event_extraction_agent.models import SourcePost

VK_API_BASE_URL = "https://api.vk.com/method"
VK_DEFAULT_API_VERSION = "5.199"
VK_MAX_WALL_GET_COUNT = 100
VK_DEFAULT_RATE_LIMIT_PER_SECOND = 20.0
VK_DEFAULT_MAX_RETRIES = 3
_VK_RETRYABLE_ERROR_CODES = {1, 6, 9, 10, 29}
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.:;!?])")
_EMOJI_RE = re.compile(
    "["
    "\U0001f1e6-\U0001f1ff"
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    "\ufe0f"
    "]+"
)


class VKApiError(RuntimeError):
    """Raised when VK API returns an error or cannot be reached."""

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        details: dict[str, Any] | None = None,
        method: str | None = None,
        source: "VKPostSource | None" = None,
        retryable: bool = False,
    ) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        self.method = method
        self.source = source
        self.retryable = retryable
        super().__init__(self._format_message())

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        method: str | None = None,
        source: "VKPostSource | None" = None,
    ) -> "VKApiError":
        error = payload.get("error", {})
        if not isinstance(error, dict):
            return cls(
                "VK API returned an unknown error",
                details=payload,
                method=method,
                source=source,
            )

        code = error.get("error_code")
        message = error.get("error_msg") or "VK API returned an error"
        safe_code = code if isinstance(code, int) else None
        return cls(
            str(message),
            code=safe_code,
            details=error,
            method=method,
            source=source,
            retryable=safe_code in _VK_RETRYABLE_ERROR_CODES,
        )

    def with_source(self, source: "VKPostSource") -> "VKApiError":
        return VKApiError(
            self.message,
            code=self.code,
            details=self.details,
            method=self.method,
            source=source,
            retryable=self.retryable,
        )

    def _format_message(self) -> str:
        parts = [self.message]
        if self.code is not None:
            parts.append(f"code={self.code}")
        if self.method:
            parts.append(f"method={self.method}")
        if self.source is not None:
            parts.append(f"source={self.source.reference}")
        return "VK API error: " + "; ".join(parts)


@dataclass(frozen=True)
class VKPostSource:
    """VK wall source accepted by VKSource."""

    owner_id: int | None = None
    domain: str | None = None
    source_name: str | None = None

    @property
    def reference(self) -> str:
        if self.owner_id is not None:
            return str(self.owner_id)
        if self.domain:
            return self.domain
        return "<unknown>"

    def to_wall_get_params(self) -> dict[str, Any]:
        if self.owner_id is not None:
            return {"owner_id": self.owner_id}
        if self.domain:
            return {"domain": self.domain}
        raise ValueError("VK source must include owner_id or domain")


@dataclass(frozen=True)
class VKFetchResult:
    """Posts fetched from VK together with per-source failures."""

    posts: list[SourcePost]
    errors: list[VKApiError]


class _RequestRateLimiter:
    def __init__(
        self,
        rate_limit_per_second: float | None,
        *,
        sleep: Any = time_module.sleep,
        monotonic: Any = time_module.monotonic,
    ) -> None:
        if rate_limit_per_second is None or rate_limit_per_second <= 0:
            self.min_interval_seconds = 0.0
        else:
            self.min_interval_seconds = 1.0 / rate_limit_per_second
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


class VKApiClient:
    """Small VK API client using only the Python standard library."""

    def __init__(
        self,
        access_token: str,
        *,
        api_version: str = VK_DEFAULT_API_VERSION,
        base_url: str = VK_API_BASE_URL,
        timeout_seconds: float = 10.0,
        rate_limit_per_second: float | None = VK_DEFAULT_RATE_LIMIT_PER_SECOND,
        max_retries: int = VK_DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = 1.0,
        sleep: Any = time_module.sleep,
        monotonic: Any = time_module.monotonic,
    ) -> None:
        if not access_token:
            raise ValueError("VK access token is required")

        self.access_token = access_token
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._sleep = sleep
        self.rate_limiter = _RequestRateLimiter(
            rate_limit_per_second,
            sleep=sleep,
            monotonic=monotonic,
        )

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        source: VKPostSource | None = None,
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        request_params["access_token"] = self.access_token
        request_params["v"] = self.api_version

        url = f"{self.base_url}/{method}?{urlencode(request_params, doseq=True)}"
        request = Request(url, headers={"User-Agent": "event-extraction-agent-vk-source/0.4"})

        last_error: VKApiError | None = None
        for attempt in range(self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                api_error = VKApiError(
                    f"HTTP request failed with status {error.code}",
                    code=error.code,
                    method=method,
                    source=source,
                    retryable=_is_retryable_http_status(error.code),
                )
                if not api_error.retryable or attempt >= self.max_retries:
                    raise api_error from error
                last_error = api_error
                self._sleep_before_retry(attempt, error)
                continue
            except URLError as error:
                api_error = VKApiError(
                    f"connection failed: {error.reason}",
                    method=method,
                    source=source,
                    retryable=True,
                )
                if attempt >= self.max_retries:
                    raise api_error from error
                last_error = api_error
                self._sleep_before_retry(attempt)
                continue
            except json.JSONDecodeError as error:
                raise VKApiError("returned invalid JSON", method=method, source=source) from error

            if "error" in payload:
                api_error = VKApiError.from_payload(payload, method=method, source=source)
                if not api_error.retryable or attempt >= self.max_retries:
                    raise api_error
                last_error = api_error
                self._sleep_before_retry(attempt)
                continue

            response_payload = payload.get("response")
            if not isinstance(response_payload, dict):
                raise VKApiError(
                    "response does not contain an object response",
                    details=payload,
                    method=method,
                    source=source,
                )

            return response_payload

        if last_error is not None:
            raise last_error
        raise VKApiError("request failed", method=method, source=source)

    def _sleep_before_retry(self, attempt: int, error: HTTPError | None = None) -> None:
        retry_after = error.headers.get("Retry-After") if error is not None and error.headers is not None else None
        if retry_after:
            try:
                delay = max(0.0, float(retry_after))
            except ValueError:
                delay = 0.0
        else:
            delay = min(30.0, self.retry_backoff_seconds * (2.0**attempt))
        if delay > 0:
            self._sleep(delay)

    def get_wall(
        self,
        source: VKPostSource,
        *,
        count: int = VK_MAX_WALL_GET_COUNT,
        offset: int = 0,
        wall_filter: str = "owner",
        extended: bool = True,
    ) -> dict[str, Any]:
        safe_count = max(0, min(count, VK_MAX_WALL_GET_COUNT))
        params = {
            **source.to_wall_get_params(),
            "count": safe_count,
            "offset": max(0, offset),
            "filter": wall_filter,
            "extended": 1 if extended else 0,
        }
        return self.call("wall.get", params, source=source)


class VKSource:
    """Production SourceAdapter for VK wall posts."""

    def __init__(
        self,
        access_token: str,
        sources: list[VKPostSource | str | int],
        *,
        posts_per_source_limit: int = VK_MAX_WALL_GET_COUNT,
        offset: int = 0,
        wall_filter: str = "owner",
        api_version: str = VK_DEFAULT_API_VERSION,
        base_url: str = VK_API_BASE_URL,
        timeout_seconds: float = 10.0,
        batch_size: int = VK_MAX_WALL_GET_COUNT,
        continue_on_source_error: bool = True,
        rate_limit_per_second: float | None = VK_DEFAULT_RATE_LIMIT_PER_SECOND,
        max_retries: int = VK_DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = 1.0,
        sleep: Any = time_module.sleep,
        monotonic: Any = time_module.monotonic,
    ) -> None:
        if not sources:
            raise ValueError("VKSource requires at least one source")

        self.sources = [parse_vk_source(source) for source in sources]
        self.posts_per_source_limit = max(0, posts_per_source_limit)
        self.offset = max(0, offset)
        self.wall_filter = wall_filter
        self.batch_size = max(1, min(batch_size, VK_MAX_WALL_GET_COUNT))
        self.continue_on_source_error = continue_on_source_error
        self.errors: list[VKApiError] = []
        self.api_client = VKApiClient(
            access_token=access_token,
            api_version=api_version,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            rate_limit_per_second=rate_limit_per_second,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep=sleep,
            monotonic=monotonic,
        )

    def fetch_posts(self, *, cached_external_ids: set[str] | None = None) -> list[SourcePost]:
        return self.fetch_posts_with_errors(cached_external_ids=cached_external_ids).posts

    def fetch_posts_with_errors(self, *, cached_external_ids: set[str] | None = None) -> VKFetchResult:
        posts: list[SourcePost] = []
        errors: list[VKApiError] = []
        cached_ids = cached_external_ids or set()
        for source in self.sources:
            try:
                posts.extend(self._fetch_source_posts(source, cached_external_ids=cached_ids))
            except VKApiError as error:
                source_error = error if error.source is not None else error.with_source(source)
                errors.append(source_error)
                if not self.continue_on_source_error:
                    self.errors = errors
                    raise source_error from error
        self.errors = errors
        return VKFetchResult(posts=posts, errors=errors)

    def _fetch_source_posts(
        self,
        source: VKPostSource,
        *,
        cached_external_ids: set[str],
    ) -> list[SourcePost]:
        current_offset = self.offset
        posts: list[SourcePost] = []
        counted_posts = 0

        while counted_posts < self.posts_per_source_limit:
            requested_count = min(self.batch_size, self.posts_per_source_limit - counted_posts)
            response = self.api_client.get_wall(
                source,
                count=requested_count,
                offset=current_offset,
                wall_filter=self.wall_filter,
                extended=True,
            )
            items = response.get("items", [])
            if not isinstance(items, list) or not items:
                break

            sources_by_owner_id = _build_sources_by_owner_id(response)
            for item in items:
                if not isinstance(item, dict):
                    continue
                post = _to_source_post(item, source, sources_by_owner_id)
                if post is not None:
                    posts.append(post)
                    if not (post.is_pinned and post.external_id in cached_external_ids):
                        counted_posts += 1
                    if counted_posts >= self.posts_per_source_limit:
                        break

            fetched_count = len(items)
            current_offset += fetched_count
            if fetched_count < requested_count:
                break

        return posts


def parse_vk_source(source: VKPostSource | str | int) -> VKPostSource:
    if isinstance(source, VKPostSource):
        return source
    if isinstance(source, int):
        return VKPostSource(owner_id=source)

    value = source.strip()
    if not value:
        raise ValueError("VK source cannot be empty")

    if value.lstrip("-").isdigit():
        return VKPostSource(owner_id=int(value))

    parsed_url = urlparse(value)
    if parsed_url.scheme and parsed_url.netloc:
        return _parse_vk_url(parsed_url)

    return _parse_vk_path(value)


def build_vk_post_url(owner_id: int, post_id: int) -> str:
    return f"https://vk.com/wall{owner_id}_{post_id}"


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code <= 599


def _to_source_post(
    item: dict[str, Any],
    source: VKPostSource,
    sources_by_owner_id: dict[int, dict[str, Any]],
) -> SourcePost | None:
    post_id = _require_int(item, "id")
    owner_id = _extract_owner_id(item, source)
    text = item.get("text") if isinstance(item.get("text"), str) else ""
    normalized_text = _clean_text(text)
    if not normalized_text:
        return None

    return SourcePost(
        text=text,
        raw_text=normalized_text,
        source_name=_resolve_source_name(owner_id, source, sources_by_owner_id),
        source_url=build_vk_post_url(owner_id, post_id),
        published_at=_unix_to_iso(item.get("date") if isinstance(item.get("date"), int) else None),
        external_id=f"vk:wall{owner_id}_{post_id}",
        is_pinned=bool(item.get("is_pinned")),
    )


def _parse_vk_url(parsed_url: Any) -> VKPostSource:
    host = parsed_url.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in {"vk.com", "m.vk.com"}:
        raise ValueError("VK URL must use vk.com or m.vk.com host")

    path = parsed_url.path.strip("/")
    if not path:
        query = parse_qs(parsed_url.query)
        wall = query.get("w", [""])[0]
        if wall.startswith("wall"):
            path = wall

    return _parse_vk_path(path)


def _parse_vk_path(path: str) -> VKPostSource:
    normalized = path.strip().strip("/")
    if not normalized:
        raise ValueError("VK source path cannot be empty")

    first_segment = normalized.split("/", 1)[0]
    if first_segment.startswith("wall"):
        wall_ref = first_segment.removeprefix("wall")
        owner_part = wall_ref.split("_", 1)[0]
        return VKPostSource(owner_id=int(owner_part))
    if first_segment.startswith("club") and first_segment[4:].isdigit():
        return VKPostSource(owner_id=-int(first_segment[4:]))
    if first_segment.startswith("public") and first_segment[6:].isdigit():
        return VKPostSource(owner_id=-int(first_segment[6:]))
    if first_segment.startswith("id") and first_segment[2:].isdigit():
        return VKPostSource(owner_id=int(first_segment[2:]))

    return VKPostSource(domain=first_segment)


def _require_int(item: dict[str, Any], field_name: str) -> int:
    value = item.get(field_name)
    if not isinstance(value, int):
        raise ValueError(f"VK post item must include integer {field_name}")
    return value


def _extract_owner_id(item: dict[str, Any], source: VKPostSource) -> int:
    owner_id = item.get("owner_id")
    if isinstance(owner_id, int):
        return owner_id
    if source.owner_id is not None:
        return source.owner_id
    raise ValueError("VK post item must include owner_id when source domain is used")


def _unix_to_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _build_sources_by_owner_id(response: dict[str, Any]) -> dict[int, dict[str, Any]]:
    sources: dict[int, dict[str, Any]] = {}

    for group in response.get("groups", []):
        if isinstance(group, dict) and isinstance(group.get("id"), int):
            sources[-group["id"]] = group

    for profile in response.get("profiles", []):
        if isinstance(profile, dict) and isinstance(profile.get("id"), int):
            sources[profile["id"]] = profile

    return sources


def _resolve_source_name(
    owner_id: int,
    source: VKPostSource,
    sources_by_owner_id: dict[int, dict[str, Any]],
) -> str | None:
    if source.source_name:
        return source.source_name

    source_payload = sources_by_owner_id.get(owner_id)
    if not source_payload:
        return source.domain

    if "name" in source_payload and isinstance(source_payload["name"], str):
        return source_payload["name"]

    first_name = source_payload.get("first_name")
    last_name = source_payload.get("last_name")
    full_name = " ".join(part for part in (first_name, last_name) if isinstance(part, str) and part.strip())
    return full_name or source.domain


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    compacted = " ".join(_EMOJI_RE.sub("", text).split())
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", compacted)
