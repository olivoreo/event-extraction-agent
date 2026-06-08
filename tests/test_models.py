from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from event_extraction_agent import AttendanceType, Event, EventStatus, EventType, SourcePost
from event_extraction_agent.models import normalize_event_type


VALID_EVENT_DATA = {
    "title": "Открытая лекция по ИИ",
    "description": "Лекция для студентов и начинающих разработчиков",
    "start_at": "2026-06-15T19:00:00+03:00",
    "end_at": None,
    "timezone": "Europe/Moscow",
    "city": "Москва",
    "venue_name": "ДК Горбунова",
    "address": None,
    "event_type": "EducationEvent",
    "attendance_type": "OfflineEventAttendanceMode",
    "event_status": "EventScheduled",
    "language": "ru",
    "source_name": "VK",
    "source_url": "https://vk.com/example?w=wall-1_1",
    "raw_text": "15 июня в 19:00 в Москве пройдет открытая лекция по ИИ.",
    "relevant_roles": ["Participant"],
    "industries": ["ITAndVideoGames"],
    "skills": ["artificial intelligence"],
    "price_text": "free",
    "target_audience_text": "студенты и начинающие разработчики",
}


def test_source_post_rejects_blank_text():
    with pytest.raises(ValidationError):
        SourcePost(text="  ")


def test_event_validates_contract_fields():
    event = Event(**VALID_EVENT_DATA)

    assert event.title == "Открытая лекция по ИИ"
    assert event.start_at == datetime(2026, 6, 15, 19, 0, tzinfo=timezone(timedelta(hours=3)))
    assert event.event_type == EventType.EDUCATION
    assert event.attendance_type == AttendanceType.OFFLINE
    assert event.event_status == EventStatus.SCHEDULED


def test_event_requires_core_fields():
    for field in ["title", "start_at", "timezone", "language", "raw_text"]:
        with pytest.raises(ValidationError):
            Event(**{**VALID_EVENT_DATA, field: None})


def test_event_has_no_persistence_fields():
    with pytest.raises(ValidationError):
        Event(**{**VALID_EVENT_DATA, "id": 1})


def test_enum_values_are_normalized_like_source_project():
    assert normalize_event_type("EducationEvent") == EventType.EDUCATION
    assert normalize_event_type("educationevent") == EventType.EDUCATION
    assert normalize_event_type("unknown") == EventType.UNKNOWN
    assert normalize_event_type("CustomFestival") == EventType.OTHER

    with pytest.raises(ValidationError):
        Event(**{**VALID_EVENT_DATA, "event_type": ""})
