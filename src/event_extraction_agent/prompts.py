from __future__ import annotations

import json
from typing import Any


EVENT_EXTRACTION_FIELDS = (
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
)

EVENT_TYPE_VALUES = (
    "EducationEvent",
    "BusinessEvent",
    "ChildrensEvent",
    "ComedyEvent",
    "CompetitionEvent",
    "CourseInstance",
    "DanceEvent",
    "DeliveryEvent",
    "ExhibitionEvent",
    "Festival",
    "FoodEvent",
    "Hackathon",
    "LiteraryEvent",
    "MusicEvent",
    "PublicationEvent",
    "SaleEvent",
    "ScreeningEvent",
    "SocialEvent",
    "SportsEvent",
    "TheaterEvent",
    "VisualArtsEvent",
)

ATTENDANCE_TYPE_VALUES = (
    "OfflineEventAttendanceMode",
    "OnlineEventAttendanceMode",
    "MixedEventAttendanceMode",
    "unknown",
)

EVENT_STATUS_VALUES = (
    "EventScheduled",
    "EventCancelled",
    "EventMovedOnline",
    "EventPostponed",
    "EventRescheduled",
    "unknown",
)

ROLE_VALUES = (
    "Organizer",
    "CoOrganizer",
    "Speaker",
    "Expert",
    "Host",
    "Participant",
    "Spectator",
    "Volunteer",
    "Partner",
    "Sponsor",
    "Media",
    "Guest",
)

INDUSTRY_VALUES = (
    "Art",
    "FolkCrafts",
    "PerformingArts",
    "FilmAndAnimation",
    "Photography",
    "Design",
    "Fashion",
    "Architecture",
    "TelevisionAndRadio",
    "Advertising",
    "Publishing",
    "ITAndVideoGames",
    "Tourism",
)

SKIP_REASONS = (
    "not_event_announcement",
    "past_event_report",
    "vacancy",
    "safety_instruction",
    "general_news",
    "media_recap",
)

SYSTEM_PROMPT = """
Ты — extraction-компонент backend-системы для анонсов мероприятий.
Верни только валидный JSON-объект без prose, markdown, пояснений и свободного текста.

Главное правило: не выдумывай фактические поля вроде даты, адреса, площадки, цены и аудитории. Если факта нет в тексте или он не задан входными метаданными, верни null.

Порядок решения:
1. Определи, является ли весь пост прямым анонсом, а не просто упоминанием события.
2. Если это анонс, определи жизненный цикл события: старт, период проведения, дедлайн регистрации/участия, дата объявления результатов, перенос/отмена.
3. Только после этого заполняй поля event. Не выбирай дату механически: сначала пойми, какую роль дата играет в тексте.

Сначала строго реши, является ли пост прямым анонсом будущего или актуального мероприятия:
- если это отчет о прошедшем событии, итоги розыгрыша, поздравление победителя, вакансия, инструкция по безопасности, новость, поздравление, подборка кадров или медиа-рекап без нового призыва прийти/зарегистрироваться, верни "is_event": false;
- если пост только упоминает дату мероприятия внутри другой темы (итоги, отчет, новость, розыгрыш, благодарность, фото/видео), это не анонс;
- если это реклама образовательной программы, поступления, набора документов, курса или сервиса без конкретной даты/периода занятия, дедлайна или события для участия, верни "is_event": false;
- если пост говорит "голосование продолжается", "приём заявок стартовал", "успей подать заявку", "дедлайн" или "регистрация до", это актуальное событие/активность участия, даже если нет офлайн-встречи;
- если пост содержит несколько самостоятельных событий с разным временем/смыслом и их нельзя честно описать одним event, верни "is_event": true с event=null или неполным event; валидация пометит такой пост invalid. Не склеивай два события в одно название через "+".
- если это прямое приглашение, напоминание, перенос/отмена или программа мероприятия для посетителей, верни "is_event": true и заполни "event".

Правила дат:
- start_at и end_at возвращай в ISO 8601 без часового offset: "2026-06-19T17:00:00", не "2026-06-19T17:00:00+03:00" и не "2026-06-19T17:00:00Z".
- Если в тексте есть день и месяц без года, используй год из published_at. Если такая дата раньше published_at, используй следующий год.
- Если дата мероприятия есть, но явного времени начала нет, используй 00:00:00. Это значит "время не указано", а не полночь как фактическое время.
- Если нет уверенной даты мероприятия, верни start_at=null. Не выдумывай дату.
- Для конкурсов, наборов, регистраций, челленджей и кампаний start_at — дата начала участия/приема заявок/периода активности. Если текст говорит, что конкурс или прием уже стартовал, но не дает отдельную дату старта, используй дату published_at с временем 00:00:00.
- Если текст говорит "успей подать заявку до", "регистрация открыта до", "дедлайн", "приём заявок продлится до" и отдельная дата старта не указана, start_at — published_at с временем 00:00:00, а указанная дата — end_at. Не ставь дедлайн в start_at.
- Дату дедлайна, окончания приема заявок, финала или объявления результатов используй как end_at, если она не является датой начала.
- Если указана длительность занятия ("1,5 часа", "2 часа"), не вычисляй end_at по длительности. end_at заполняй только если конечная дата/время явно написаны.
- Для многодневного мероприятия start_at — первая дата/время, end_at — последняя дата/время, если она явно указана.
- Форматы "6 МАРТА | 18:00", "7 марта | 15:30", "1 июня в 18.30" и "5 июня в 18:00" являются достаточными для start_at.

Правила полей:
- source_name и source_url бери из метаданных, если они переданы.
- raw_text должен точно содержать очищенный текст поста, переданный на вход.
- title всегда должен быть строкой для is_event=true; если отдельного названия нет, сформулируй короткий title из первых слов анонса.
- language для русскоязычных постов возвращай "ru".
- description — короткое описание события, а не полный пост.
- venue_name, address, city и target_audience_text заполняй только при явном наличии.
- timezone возвращай как IANA identifier, например "Europe/Moscow", если он явно указан или разумно следует из города/локального контекста; иначе возвращай "unknown".
- event_type, attendance_type и event_status должны быть выбраны только из разрешенных словарей.
- Для event_type не возвращай "unknown": если is_event=true, выбери самый близкий тип из списка.
- Если формат участия не указан, возвращай "OfflineEventAttendanceMode".
- Если нет явной отмены, переноса, смены даты или перевода в онлайн, возвращай "EventScheduled".
- relevant_roles возвращай только на английском из ROLE_VALUES.
- industries возвращай только из INDUSTRY_VALUES; не возвращай типы мероприятий и произвольные категории.
- skills можно возвращать как короткие английские labels, если в тексте явно есть навыки или активности; иначе null.
- price_text возвращай как "free", если цена не указана, вход свободный или участие бесплатное. Если указана цена или билеты, верни короткое описание вроде "1600 RUB", "tickets required" или "tickets: https://example.com".

Подсказки для выбора ближайшего event_type:
- лекция, вебинар, мастер-класс, интенсив, курс -> EducationEvent или CourseInstance;
- конкурс, соревнование, финал, отбор, состязание -> CompetitionEvent;
- концерт, музыкальный вечер, выступление группы -> MusicEvent;
- спектакль, постановка, театральное представление -> TheaterEvent;
- кинопоказ, фильм, премьера фильма -> ScreeningEvent;
- выставка, экспозиция, арт-показ -> ExhibitionEvent;
- гастро-вечер, дегустация, ресторанный вечер, кулинария -> FoodEvent;
- хакатон, командная разработка прототипов -> Hackathon;
- деловая встреча, бизнес-завтрак, конференция предпринимателей -> BusinessEvent;
- детский праздник или событие специально для детей -> ChildrensEvent;
- спортивный матч, тренировка, забег, турнир по спорту -> SportsEvent.
""".strip()


def response_schema() -> dict[str, Any]:
    event_schema: dict[str, Any] = {field: None for field in EVENT_EXTRACTION_FIELDS}
    event_schema["event_type"] = list(EVENT_TYPE_VALUES)
    event_schema["attendance_type"] = list(ATTENDANCE_TYPE_VALUES)
    event_schema["event_status"] = list(EVENT_STATUS_VALUES)
    event_schema["relevant_roles"] = list(ROLE_VALUES)
    event_schema["industries"] = list(INDUSTRY_VALUES)
    return {
        "is_event": True,
        "skip_reason": None,
        "event": event_schema,
    }


def build_extraction_prompt(
    raw_text: str,
    source_name: str | None = None,
    source_url: str | None = None,
    published_at: str | None = None,
    external_id: str | None = None,
    current_datetime: str | None = None,
) -> str:
    source_metadata = {
        "source_name": source_name,
        "source_url": source_url,
        "published_at": published_at,
        "current_datetime": current_datetime,
        "external_id": external_id,
    }

    return f"""
Извлеки структуру события из одного текстового поста.

Ожидаемый JSON:
{json.dumps(response_schema(), ensure_ascii=False, indent=2)}

Разрешенные event_type:
{json.dumps(EVENT_TYPE_VALUES, ensure_ascii=False)}

Разрешенные attendance_type:
{json.dumps(ATTENDANCE_TYPE_VALUES, ensure_ascii=False)}

Разрешенные event_status:
{json.dumps(EVENT_STATUS_VALUES, ensure_ascii=False)}

Разрешенные relevant_roles:
{json.dumps(ROLE_VALUES, ensure_ascii=False)}

Разрешенные industries:
{json.dumps(INDUSTRY_VALUES, ensure_ascii=False)}

Допустимые skip_reason для is_event=false:
{json.dumps(SKIP_REASONS, ensure_ascii=False)}

Метаданные источника:
{json.dumps(source_metadata, ensure_ascii=False, indent=2)}

Очищенный текст поста:
{raw_text}
""".strip()


EVENT_TYPE_CLASSIFICATION_PROMPT = """
Ты классифицируешь тип мероприятия по смыслу текста.
Верни только JSON без markdown и пояснений.
Выбери ровно один event_type из разрешенного списка.
SocialEvent используй только для общих социальных/общественных событий, которые не подходят под более конкретные типы.
""".strip()


def build_event_type_classification_prompt(raw_text: str, draft: dict[str, Any] | None = None) -> str:
    return f"""
Разрешенные event_type:
{json.dumps(EVENT_TYPE_VALUES, ensure_ascii=False)}

Подсказки:
- конкурс, соревнование, финал, отбор, состязание -> CompetitionEvent;
- лекция, вебинар, мастер-класс, интенсив, курс -> EducationEvent или CourseInstance;
- концерт, музыкальный вечер, выступление группы -> MusicEvent;
- спектакль, постановка, театральное представление -> TheaterEvent;
- кинопоказ, фильм, премьера фильма -> ScreeningEvent;
- выставка, экспозиция, арт-показ -> ExhibitionEvent;
- гастро-вечер, дегустация, ресторанный вечер, кулинария -> FoodEvent;
- хакатон, командная разработка прототипов -> Hackathon;
- деловая встреча, бизнес-завтрак, конференция предпринимателей -> BusinessEvent;
- детский праздник или событие специально для детей -> ChildrensEvent;
- спортивный матч, тренировка, забег, турнир по спорту -> SportsEvent.

Черновик события:
{json.dumps(draft or {}, ensure_ascii=False, indent=2)}

Текст:
{raw_text}

Ответ:
{{"event_type": "<one allowed event_type>"}}
""".strip()
