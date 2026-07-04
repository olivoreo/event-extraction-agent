from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from event_extraction_agent import (
    AttendanceType,
    BatchExtractionResult,
    Event,
    EventType,
    ExtractionOutcome,
    ExtractionStatus,
    SourcePost,
    drop_event,
)
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


def test_source_post_uses_raw_text_for_prompt_when_present():
    post = SourcePost(text=" Исходный\nтекст ", raw_text=" Очищенный текст ")

    assert post.text == "Исходный\nтекст"
    assert post.raw_text == "Очищенный текст"
    assert post.raw_text_for_prompt() == "Очищенный текст"
    assert SourcePost(text="Только текст").raw_text_for_prompt() == "Только текст"


def test_event_validates_contract_fields():
    event = Event(**VALID_EVENT_DATA)

    assert event.title == "Открытая лекция по ИИ"
    assert event.start_at == datetime(2026, 6, 15, 19, 0, tzinfo=timezone(timedelta(hours=3)))
    assert event.event_type == EventType.EDUCATION
    assert event.attendance_type == AttendanceType.OFFLINE


def test_event_requires_core_fields():
    for field in ["title", "timezone", "language", "raw_text"]:
        with pytest.raises(ValidationError):
            Event(**{**VALID_EVENT_DATA, field: None})


def test_event_allows_missing_start_at():
    event = Event(**{**VALID_EVENT_DATA, "start_at": None, "end_at": "2026-06-20T00:00:00"})

    assert event.start_at is None


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


def test_batch_extraction_result_can_be_saved_and_loaded_for_incremental_processing(tmp_path):
    post = SourcePost(
        text="15 июня в 19:00 пройдет лекция.",
        raw_text="15 июня в 19:00 пройдет лекция.",
        source_name="VK",
        source_url="https://vk.com/example?w=wall-1_1",
        published_at=datetime(2026, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))),
        external_id="vk:-1_1",
    )
    outcome = ExtractionOutcome(
        status=ExtractionStatus.EXTRACTED,
        event=Event(**{**VALID_EVENT_DATA, "raw_text": post.raw_text}),
        post=post,
        raw_llm_metadata={"model": "test-model"},
    )
    result = BatchExtractionResult.from_outcomes([outcome])
    path = tmp_path / "events_result.json"

    result.save_json(path)
    loaded = BatchExtractionResult.load_json(path)

    assert loaded == result
    assert loaded.outcomes[0].post.external_id == "vk:-1_1"
    assert loaded.outcomes[0].post.raw_text_for_prompt() == "15 июня в 19:00 пройдет лекция."


def test_drop_event_removes_one_event_from_multi_event_outcome():
    post = SourcePost(text="15 и 16 июня пройдут лекции.")
    first_event = Event(**VALID_EVENT_DATA)
    second_event = Event(**{**VALID_EVENT_DATA, "title": "Вторая лекция", "start_at": "2026-06-16T19:00:00+03:00"})
    result = BatchExtractionResult.from_outcomes(
        [
            ExtractionOutcome(
                status=ExtractionStatus.EXTRACTED,
                event=None,
                events=[first_event, second_event],
                post=post,
            )
        ]
    )

    updated = drop_event(result, outcome_index=0, event_index=0)

    assert updated.total == 1
    assert updated.extracted == 1
    assert updated.skipped == 0
    assert updated.outcomes[0].event == second_event
    assert updated.outcomes[0].events == [second_event]
    assert [(item.outcome_index, item.event_index, item.event.title) for item in updated.events] == [
        (0, 0, "Вторая лекция")
    ]


def test_drop_event_marks_outcome_skipped_when_last_event_is_removed():
    post = SourcePost(text="15 июня в 19:00 пройдет лекция.")
    event = Event(**VALID_EVENT_DATA)
    result = BatchExtractionResult.from_outcomes(
        [ExtractionOutcome(status=ExtractionStatus.EXTRACTED, event=event, post=post)]
    )

    updated = drop_event(result, outcome_index=0)

    assert updated.total == 1
    assert updated.extracted == 0
    assert updated.skipped == 1
    assert updated.events == []
    assert updated.outcomes[0].status == ExtractionStatus.SKIPPED
    assert updated.outcomes[0].event is None
    assert updated.outcomes[0].events is None
    assert updated.outcomes[0].errors[-1].code == "dropped_by_user"
