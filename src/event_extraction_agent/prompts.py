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
    "language",
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
Ты агент структурирования анонсов мероприятий. Верни только валидный JSON без markdown/prose.
Точные факты — даты, время, название, город, площадка, адрес, цена и ссылки — бери только из текста/метаданных. Не выдумывай их. Смысловые поля — description, event_type, attendance_type, target_audience_text, relevant_roles, industries и skills — заполняй по смыслу анонса, даже если они не названы дословно.

Сначала реши, есть ли прямой анонс будущего/актуального события или периода участия. Отчеты, итоги, вакансии, новости, поздравления, медиа-рекапы и реклама обучения без даты/периода/дедлайна -> is_event=false. Личные истории, мнения, интервью или планы автора без публичного приглашения/регистрации/расписания -> is_event=false. Голосование, прием заявок, регистрация, дедлайн, конкурсный период -> is_event=true как активность участия.

Если одно событие - заполни event, events=null/omit. Если несколько самостоятельных событий или непоследовательных повторов - event=null/omit, events отдельными объектами. Не склеивай события в один title.

Даты: start_at/end_at в ISO без offset/Z. Если нет ни даты/времени начала, ни даты окончания/дедлайна, это не мероприятие -> is_event=false. День+месяц без года бери из published_at; если дата раньше published_at, используй следующий год. Нет времени -> 00:00:00. Нет уверенной даты начала -> start_at=null. Не подменяй дату события дедлайном/розыгрышем/итогами. Для регистрации/заявок/голосования дедлайн без старта -> start_at=null, end_at=дедлайн; период "с X по Y" -> start_at=X, end_at=Y. end_at только при явном окончании этого же объекта. Длительность ("2 часа") не вычисляй. Последовательные дни можно одним периодом; непоследовательные даты - отдельные events.

Поля: title строка для is_event=true; language для русского "ru"; timezone IANA если ясно из города/контекста, иначе "unknown"; attendance_type default OfflineEventAttendanceMode; price_text "free" если бесплатно/цена не указана, иначе кратко. venue_name/address/city только из явных фактов. Не возвращай source_name, source_url и raw_text.
title — только название мероприятия без приписок и описательных деталей. Если название явно выделено кавычками/капсом/отдельной строкой, бери только выделенное название. Если явного названия нет, title — краткая суть события из текста.
description — самодостаточная сухая выжимка, которая полностью раскрывает смысл мероприятия. Перескажи факты нейтрально: не копируй рекламные абзацы, убери эмоции, оценочные прилагательные, призывы, приветствия, хештеги и повторы. Сохрани все существенное: формат, темы/программу, условия участия, дедлайны, льготы/ограничения и контакты. Все указанные в посте ссылки для регистрации, покупки билетов, подачи заявки или участия перенеси в description дословно. Не ограничивай description одним предложением и не дублируй только title.
target_audience_text — непустой понятный список подходящих групп и профессиональных ролей только на русском языке. Включай как явно указанную аудиторию, так и очевидно подходящие по теме и формату группы; например, для концерта — зрители и любители соответствующей музыки, для IT-форума — подходящие IT-специалисты и учащиеся. null допустим только когда аудиторию нельзя разумно определить даже по смыслу события. Не добавляй случайные аудитории.

Категории выбирай только из разрешенных значений в пользовательском prompt. Для event_type выбери ближайший тип, не unknown. relevant_roles — массив всех подходящих способов участия в событии из ROLE_VALUES, их может быть несколько. Participant/Spectator добавляй, когда люди могут участвовать/смотреть; Volunteer/Speaker/Organizer и другие специальные роли — только когда пост предлагает или подразумевает именно такой способ участия. Не добавляй роль лишь потому, что такой человек присутствует в описании события. industries только INDUSTRY_VALUES, skills — короткие английские labels или null.
Подсказки event_type: лекция/вебинар/мастер-класс/интенсив/курс -> EducationEvent/CourseInstance; конкурс/соревнование/финал/отбор -> CompetitionEvent; концерт -> MusicEvent; спектакль -> TheaterEvent; кино -> ScreeningEvent; выставка -> ExhibitionEvent; гастро/кулинария -> FoodEvent; хакатон -> Hackathon; деловая встреча/конференция -> BusinessEvent; детское -> ChildrensEvent; спорт -> SportsEvent.

Перед ответом проверь каждый event: description содержит сухие факты вместо рекламного пересказа и сохраняет полезные ссылки; target_audience_text заполнен по смыслу на русском; relevant_roles содержит все и только подходящие способы участия.
""".strip()


def response_schema() -> dict[str, Any]:
    event_schema: dict[str, Any] = {field: None for field in EVENT_EXTRACTION_FIELDS}
    event_schema["event_type"] = list(EVENT_TYPE_VALUES)
    event_schema["attendance_type"] = list(ATTENDANCE_TYPE_VALUES)
    event_schema["relevant_roles"] = list(ROLE_VALUES)
    event_schema["industries"] = list(INDUSTRY_VALUES)
    return {
        "is_event": True,
        "skip_reason": None,
        "event": event_schema,
        "events": ["same shape as event"],
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
JSON shape:
{json.dumps(response_schema(), ensure_ascii=False, separators=(",", ":"))}

skip_reason values: {json.dumps(SKIP_REASONS, ensure_ascii=False, separators=(",", ":"))}
metadata: {json.dumps(source_metadata, ensure_ascii=False, separators=(",", ":"))}
raw_text: {raw_text}
""".strip()


def build_invalid_date_repair_prompt(
    raw_text: str,
    previous_errors: list[dict[str, str]],
    source_name: str | None = None,
    source_url: str | None = None,
    published_at: str | None = None,
    external_id: str | None = None,
    current_datetime: str | None = None,
) -> str:
    return f"""
Предыдущий ответ не прошел backend validation: {json.dumps(previous_errors, ensure_ascii=False, separators=(",", ":"))}
Исправь только даты мероприятия. end_at не может быть раньше start_at.
Если в тексте нет корректной даты окончания/дедлайна, верни end_at=null.
Если после проверки нет ни start_at, ни end_at, верни is_event=false.

{build_extraction_prompt(
    raw_text=raw_text,
    source_name=source_name,
    source_url=source_url,
    published_at=published_at,
    external_id=external_id,
    current_datetime=current_datetime,
)}
""".strip()


EVENT_TYPE_CLASSIFICATION_PROMPT = """
Ты классифицируешь тип мероприятия по смыслу текста.
Верни только JSON без markdown и пояснений.
Выбери ровно один event_type из разрешенного списка.
SocialEvent используй только для общих социальных/общественных событий, которые не подходят под более конкретные типы.
""".strip()


TITLE_DESCRIPTION_REFINEMENT_PROMPT = """
Ты исправляешь только title и description события по тексту поста.
Верни только JSON без markdown и пояснений.
title должен быть названием мероприятия, а не рекламной фразой, темой поста или чужим событием.
Если название явно выделено кавычками/капсом/отдельной строкой, бери только выделенное название.
Если явного названия нет, title — краткая суть события из текста.
description — самодостаточная сухая выжимка без рекламной воды. Сохрани формат, темы/программу, условия участия, дедлайны, льготы/ограничения, контакты и дословно все ссылки для регистрации, билетов, заявок или участия. Не ограничивай одним предложением и не дублируй только title.
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


def build_title_description_refinement_prompt(raw_text: str, draft: dict[str, Any] | None = None) -> str:
    return f"""
Черновик события:
{json.dumps(draft or {}, ensure_ascii=False, indent=2)}

Текст:
{raw_text}

Ответ:
{{"title": "<event title>", "description": "<complete concise event description or null>"}}
""".strip()
