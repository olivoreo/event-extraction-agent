from __future__ import annotations

from datetime import datetime
from enum import StrEnum
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


class EventStatus(StrEnum):
    SCHEDULED = "EventScheduled"
    CANCELLED = "EventCancelled"
    MOVED_ONLINE = "EventMovedOnline"
    POSTPONED = "EventPostponed"
    RESCHEDULED = "EventRescheduled"
    UNKNOWN = "unknown"
    OTHER = "other"


class ExtractionStatus(StrEnum):
    EXTRACTED = "extracted"
    SKIPPED = "skipped"
    INVALID = "invalid"
    LLM_ERROR = "llm_error"


_EVENT_TYPE_ALIASES = {value.value.lower(): value for value in EventType}
_ATTENDANCE_TYPE_ALIASES = {value.value.lower(): value for value in AttendanceType}
_EVENT_STATUS_ALIASES = {value.value.lower(): value for value in EventStatus}


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


def normalize_event_status(value: Any) -> EventStatus:
    return _normalize_required_category(value, _EVENT_STATUS_ALIASES)  # type: ignore[return-value]


class SourcePost(BaseModel):
    """Input post shape accepted by the extraction agent."""

    model_config = ConfigDict(extra="forbid")

    text: str
    source_name: str | None = None
    source_url: str | None = None
    published_at: datetime | str | None = None
    external_id: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("text must not be blank")
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


class Event(BaseModel):
    """Canonical event model returned by this package."""

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    title: str
    description: str | None = None
    start_at: datetime
    end_at: datetime | None = None
    timezone: str
    city: str | None = None
    venue_name: str | None = None
    address: str | None = None
    event_type: EventType = EventType.UNKNOWN
    attendance_type: AttendanceType = AttendanceType.UNKNOWN
    event_status: EventStatus = EventStatus.SCHEDULED
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

    @field_validator("event_status", mode="before")
    @classmethod
    def validate_event_status(cls, value: Any) -> EventStatus:
        return normalize_event_status(value)

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
    post: SourcePost
    errors: list[ExtractionError] = Field(default_factory=list)
    raw_llm_metadata: dict[str, Any] | None = None
