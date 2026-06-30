from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(StrEnum):
    EDUCATION = "EducationEvent"
    BUSINESS = "BusinessEvent"
    CHILDRENS = "ChildrensEvent"
    COMEDY = "ComedyEvent"
    COMPETITION = "CompetitionEvent"
    COURSE_INSTANCE = "CourseInstance"
    DANCE = "DanceEvent"
    DELIVERY = "DeliveryEvent"
    EXHIBITION = "ExhibitionEvent"
    FESTIVAL = "Festival"
    FOOD = "FoodEvent"
    HACKATHON = "Hackathon"
    LITERARY = "LiteraryEvent"
    MUSIC = "MusicEvent"
    PUBLICATION = "PublicationEvent"
    SALE = "SaleEvent"
    SCREENING = "ScreeningEvent"
    SOCIAL = "SocialEvent"
    SPORTS = "SportsEvent"
    THEATER = "TheaterEvent"
    VISUAL_ARTS = "VisualArtsEvent"
    UNKNOWN = "unknown"
    OTHER = "other"


class AttendanceType(StrEnum):
    OFFLINE = "OfflineEventAttendanceMode"
    ONLINE = "OnlineEventAttendanceMode"
    MIXED = "MixedEventAttendanceMode"
    UNKNOWN = "unknown"
    OTHER = "other"


class ExtractionStatus(StrEnum):
    EXTRACTED = "extracted"
    SKIPPED = "skipped"
    INVALID = "invalid"
    LLM_ERROR = "llm_error"


class BatchMode(StrEnum):
    SEQUENTIAL = "sequential"


class ExtractionAgentConfig(BaseModel):
    """Runtime configuration for LLM clients and refinement behavior."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    main_client: Any | None = None
    refinement_client: Any | None = None
    use_event_type_refinement: bool = False
    current_datetime: str | None = None
    min_request_interval_seconds: float = 0.0
    max_retries: int = 0

    @field_validator("main_client", "refinement_client")
    @classmethod
    def client_must_implement_complete(cls, value: Any | None) -> Any | None:
        if value is not None and not callable(getattr(value, "complete", None)):
            raise ValueError("LLM client must implement complete(system_prompt, user_prompt)")
        return value

    @field_validator("min_request_interval_seconds")
    @classmethod
    def rate_limit_must_not_be_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("min_request_interval_seconds must not be negative")
        return value

    @field_validator("max_retries")
    @classmethod
    def max_retries_must_not_be_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must not be negative")
        return value


_EVENT_TYPE_ALIASES = {value.value.lower(): value for value in EventType}
_ATTENDANCE_TYPE_ALIASES = {value.value.lower(): value for value in AttendanceType}


def _normalize_required_category(value: Any, aliases: dict[str, StrEnum]) -> StrEnum:
    if isinstance(value, StrEnum):
        return value
    if value is None:
        return aliases["unknown"]
    if not isinstance(value, str):
        return aliases["other"]

    normalized = value.strip()
    if not normalized:
        raise ValueError("categorical value must not be blank")
    return aliases.get(normalized.lower(), aliases["other"])


def normalize_event_type(value: Any) -> EventType:
    return _normalize_required_category(value, _EVENT_TYPE_ALIASES)  # type: ignore[return-value]


def normalize_attendance_type(value: Any) -> AttendanceType:
    return _normalize_required_category(value, _ATTENDANCE_TYPE_ALIASES)  # type: ignore[return-value]


class SourcePost(BaseModel):
    """Input post shape accepted by the extraction agent."""

    model_config = ConfigDict(extra="forbid")

    text: str
    raw_text: str | None = None
    source_name: str | None = None
    source_url: str | None = None
    published_at: datetime | str | None = None
    external_id: str | None = None

    @field_validator("text", "raw_text")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{info.field_name} must not be blank")
        return normalized

    @field_validator("source_name", "source_url", "external_id")
    @classmethod
    def optional_text_blank_becomes_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    def published_at_for_prompt(self) -> str | None:
        if self.published_at is None:
            return None
        if isinstance(self.published_at, datetime):
            return self.published_at.isoformat()
        return self.published_at

    def raw_text_for_prompt(self) -> str:
        return self.raw_text or self.text


class Event(BaseModel):
    """Canonical event model returned by this package."""

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    title: str
    description: str | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    timezone: str
    city: str | None = None
    venue_name: str | None = None
    address: str | None = None
    event_type: EventType = EventType.UNKNOWN
    attendance_type: AttendanceType = AttendanceType.UNKNOWN
    language: str
    source_name: str | None = None
    source_url: str | None = None
    raw_text: str
    relevant_roles: list[str] | None = None
    industries: list[str] | None = None
    skills: list[str] | None = None
    price_text: str = "free"
    target_audience_text: str | None = None

    @field_validator("title", "timezone", "language", "raw_text")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("required text field must not be blank")
        return normalized

    @field_validator(
        "description",
        "city",
        "venue_name",
        "address",
        "source_name",
        "source_url",
        "price_text",
        "target_audience_text",
    )
    @classmethod
    def optional_text_blank_becomes_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("event_type", mode="before")
    @classmethod
    def validate_event_type(cls, value: Any) -> EventType:
        return normalize_event_type(value)

    @field_validator("attendance_type", mode="before")
    @classmethod
    def validate_attendance_type(cls, value: Any) -> AttendanceType:
        return normalize_attendance_type(value)

    @field_validator("end_at")
    @classmethod
    def end_at_must_not_be_before_start_at(cls, value: datetime | None, info: Any) -> datetime | None:
        if value is None:
            return None
        start_at = info.data.get("start_at")
        if start_at is not None and value < start_at:
            raise ValueError("end_at must not be before start_at")
        return value


class ExtractionError(BaseModel):
    field: str
    code: str
    message: str


class ExtractionOutcome(BaseModel):
    """Structured result of processing one source post."""

    status: ExtractionStatus
    event: Event | None
    events: list[Event] | None = None
    post: SourcePost
    errors: list[ExtractionError] = Field(default_factory=list)
    raw_llm_metadata: dict[str, Any] | None = None


class BatchExtractionSettings(BaseModel):
    """Settings for deterministic sequential batch extraction."""

    model_config = ConfigDict(extra="forbid")

    mode: BatchMode = BatchMode.SEQUENTIAL
    max_errors: int | None = None
    skip_empty: bool = True
    skip_duplicates: bool = True
    skip_event_duplicates: bool = True

    @field_validator("max_errors")
    @classmethod
    def max_errors_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("max_errors must be greater than 0")
        return value


class BatchExtractionResult(BaseModel):
    """Structured result of processing a list of source posts."""

    settings: BatchExtractionSettings = Field(default_factory=BatchExtractionSettings)
    outcomes: list[ExtractionOutcome] = Field(default_factory=list)
    total: int = 0
    extracted: int = 0
    skipped: int = 0
    invalid: int = 0
    llm_errors: int = 0
    cached: int = 0
    processed: int = 0
    error_count: int = 0
    error_limit_reached: bool = False

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        """Persist the batch result as UTF-8 JSON."""

        Path(path).write_text(self.model_dump_json(indent=indent) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "BatchExtractionResult":
        """Load a previously persisted batch result from UTF-8 JSON."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_outcomes(
        cls,
        outcomes: list[ExtractionOutcome],
        settings: BatchExtractionSettings | None = None,
        error_limit_reached: bool = False,
    ) -> "BatchExtractionResult":
        extracted = _count_status(outcomes, ExtractionStatus.EXTRACTED)
        skipped = _count_status(outcomes, ExtractionStatus.SKIPPED)
        invalid = _count_status(outcomes, ExtractionStatus.INVALID)
        llm_errors = _count_status(outcomes, ExtractionStatus.LLM_ERROR)
        cached = sum(1 for outcome in outcomes if (outcome.raw_llm_metadata or {}).get("incremental_cached") is True)
        return cls(
            settings=settings or BatchExtractionSettings(),
            outcomes=outcomes,
            total=len(outcomes),
            extracted=extracted,
            skipped=skipped,
            invalid=invalid,
            llm_errors=llm_errors,
            cached=cached,
            processed=len(outcomes) - cached,
            error_count=invalid + llm_errors,
            error_limit_reached=error_limit_reached,
        )


def _count_status(outcomes: list[ExtractionOutcome], status: ExtractionStatus) -> int:
    return sum(1 for outcome in outcomes if outcome.status == status)
