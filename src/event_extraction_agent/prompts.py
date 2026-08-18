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

Сначала реши, есть ли прямой анонс будущего/актуального события или конкретного периода участия. Отчеты, итоги, вакансии, новости, поздравления, медиа-рекапы и реклама обучения без даты/периода/дедлайна -> is_event=false. Личные истории, мнения, интервью или планы автора без публичного приглашения/регистрации/расписания -> is_event=false. Период приема заявок, регистрации, голосования или участия с конкретной датой, периодом или дедлайном можно считать event-активностью. Само упоминание регистрации, заявки, голосования или конкурса без конкретной активности и срока не делает текст событием.

Если одно событие - заполни event, events=null/omit. Если несколько самостоятельных событий или одна активность повторяется в перечисленные даты - event=null/omit, events отдельными объектами. Не склеивай события в один title.

Даты и время. Работай в таком порядке:
1. Определи, что дано в тексте: одна дата, явный диапазон дат или перечисление отдельных дат.
2. Если невозможно определить ни дату начала, ни дату окончания/дедлайн, верни is_event=false. Одного времени без даты недостаточно.
3. start_at/end_at возвращай в ISO без offset/Z. День+месяц без года бери из published_at; если получившаяся дата раньше published_at, используй следующий год.
4. Используй только дату и время, которые прямо следуют из текста. Если дата известна, но время для этой даты не указано, используй 00:00:00. Не переноси время с одной даты на другую и не вычисляй end_at из длительности вроде "2 часа".
5. Явный диапазон дат ("с 10 по 12 июня", "10–12 июня", "31 августа — 1 сентября") -> один event: start_at=начало диапазона, end_at=конец диапазона. Если время конечной даты не указано, end_at использует 00:00:00. Пример: "с 10 по 12 июня, начало 10 июня в 18:00" -> start_at=10 июня 18:00, end_at=12 июня 00:00.
6. Перечисление дат через "и" или запятую -> отдельный event на каждую дату, даже если даты идут подряд. Пример: "19 и 20 июня в 17:00" -> два events, 19 июня 17:00 и 20 июня 17:00; у каждого end_at=null, если отдельное время окончания не указано. "9 и 11 мая" -> два events. Не превращай перечисление в период без явного признака диапазона ("с ... по ..." или тире между датами).
7. end_at указывай только при явном окончании этого же event. Явное время окончания, например "с 18:00 до 21:00", можно использовать как end_at. Время начала в каждый из перечисленных дней не является временем окончания последнего дня.
8. Если уверенной даты начала нет, start_at=null. Не подменяй дату события дедлайном, розыгрышем или датой итогов. Для периода регистрации/заявок/голосования с известным только дедлайном: start_at=null, end_at=дедлайн.

Поля: title строка для is_event=true; language для русского "ru"; timezone IANA если ясно из города/контекста, иначе "unknown"; attendance_type определяй по смыслу текста: OnlineEventAttendanceMode для явно онлайн-формата, MixedEventAttendanceMode для смешанного формата, иначе используй OfflineEventAttendanceMode как значение по умолчанию. price_text "free" только если бесплатное участие указано явно, иначе точная цена/условие из текста или null. venue_name/address/city только из явных фактов. Не возвращай source_name, source_url и raw_text.
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


def extraction_json_schema() -> dict[str, Any]:
    nullable_string = {"type": ["string", "null"]}
    event_properties: dict[str, Any] = {
        "title": {"type": "string"},
        "description": nullable_string,
        "start_at": nullable_string,
        "end_at": nullable_string,
        "timezone": {"type": "string"},
        "city": nullable_string,
        "venue_name": nullable_string,
        "address": nullable_string,
        "event_type": {"type": "string", "enum": list(EVENT_TYPE_VALUES)},
        "attendance_type": {"type": "string", "enum": list(ATTENDANCE_TYPE_VALUES)},
        "language": {"type": "string"},
        "relevant_roles": {
            "anyOf": [
                {"type": "array", "items": {"type": "string", "enum": list(ROLE_VALUES)}},
                {"type": "null"},
            ]
        },
        "industries": {
            "anyOf": [
                {"type": "array", "items": {"type": "string", "enum": list(INDUSTRY_VALUES)}},
                {"type": "null"},
            ]
        },
        "skills": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "price_text": nullable_string,
        "target_audience_text": nullable_string,
    }
    event = {
        "type": "object",
        "properties": event_properties,
        "required": list(EVENT_EXTRACTION_FIELDS),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "is_event": {"type": "boolean"},
            "skip_reason": {
                "anyOf": [
                    {"type": "string", "enum": list(SKIP_REASONS)},
                    {"type": "null"},
                ]
            },
            "event": {"anyOf": [event, {"type": "null"}]},
            "events": {"anyOf": [{"type": "array", "items": event}, {"type": "null"}]},
        },
        "required": ["is_event", "skip_reason", "event", "events"],
        "additionalProperties": False,
    }


def event_type_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"event_type": {"type": "string", "enum": list(EVENT_TYPE_VALUES)}},
        "required": ["event_type"],
        "additionalProperties": False,
    }


def title_description_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": ["string", "null"]},
        },
        "required": ["title", "description"],
        "additionalProperties": False,
    }


DUPLICATE_EVENT_MERGE_PROMPT = """
Ты объединяешь смысловую информацию о двух или нескольких версиях одного и того же мероприятия.
Самая свежая версия уже выбрана системой и остается основной. Не меняй фактические и идентифицирующие поля свежего события: title, start_at, end_at, timezone, city, venue_name, address, event_type, attendance_type, language и price_text.

Верни JSON ровно с пятью ключами: description, relevant_roles, industries, skills, target_audience_text. Не добавляй другие ключи.

Алгоритм merge:
1. Самая свежая версия — основной источник истины. Сохрани всю её полезную информацию.
2. Мысленно разбей предыдущие description на отдельные атомарные факты: программа, тема, аудитория, URL, факт регистрации, дедлайн, дата, время, цена, место и т.д. Не оценивай целое предложение как одну неделимую деталь.
3. Для каждого атомарного факта из предыдущей версии реши отдельно: он дополняет fresh, конфликтует с fresh или относится к изменяемой логистике.
4. Непротиворечивые смысловые факты обязательно сохраняй: программа, темы, содержание, аудитория, URL, сам факт/способ/условие регистрации или участия и другие полезные детали. Если URL есть только в previous и fresh не содержит другой конфликтующей ссылки, этот URL должен остаться в итоговом description. Отсутствие детали в fresh само по себе не означает, что она устарела.
5. Конкретные даты, время, место, адрес, цена, формат и телефоны из previous не являются enrichment-данными. Не переноси такие старые значения, если этих же значений нет в fresh. Дедлайн регистрации — тоже конкретная дата и подчиняется этому правилу.
6. Если одно старое предложение содержит и полезную деталь, и изменяемое значение, раздели их по смыслу: полезную деталь сохрани, старое значение отбрось. Например URL рядом со старой датой/ценой нужно рассматривать отдельно от даты/цены.
7. При любом конфликте или явной замене информации всегда доверяй fresh. Если fresh содержит новую ссылку или новые условия регистрации, используй только fresh-версию.
8. Не выдумывай факты, ссылки или условия, которых нет ни в одной переданной версии события.
9. Перед ответом проверь ссылки отдельно: каждая полезная ссылка на билеты, регистрацию, заявку или участие из previous должна либо остаться в итоговом description, либо быть отброшена только потому, что fresh содержит новую ссылку того же назначения. Если fresh не содержит заменяющей ссылки, не теряй такую ссылку из previous.
10. description должен быть единым сухим самодостаточным описанием без рекламы, повторов и упоминаний того, что информация была объединена.
11. relevant_roles, industries и skills объединяй без дублей. Используй только разрешенные значения для relevant_roles и industries.
12. target_audience_text объедини по смыслу кратко и на русском языке, без случайных аудиторий.
13. null используй только когда полезной информации для поля нет ни в одной версии.

Примеры:
- fresh: "Напоминание о концерте."; previous: "Билеты: https://old.example/tickets" -> сохрани ссылку из previous, потому что fresh ей не противоречит.
- fresh: "Билеты: https://new.example/tickets"; previous: "Билеты: https://old.example/tickets" -> сохрани только https://new.example/tickets, потому что свежая версия заменила старую.
- fresh: "Концерт 12 сентября, билет 900 ₽."; previous: "10 сентября, билет 700 ₽: https://old.example/tickets. Регистрация обязательна до 8 сентября." -> итоговый смысл description: "Концерт 12 сентября, билет 900 ₽. Регистрация обязательна. Билеты: https://old.example/tickets". Не переноси 10 сентября, 700 ₽ или дедлайн 8 сентября.

Верни только валидный JSON без markdown/prose.
""".strip()


def duplicate_event_merge_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "description": {"type": ["string", "null"]},
            "relevant_roles": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string", "enum": list(ROLE_VALUES)}},
                    {"type": "null"},
                ]
            },
            "industries": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string", "enum": list(INDUSTRY_VALUES)}},
                    {"type": "null"},
                ]
            },
            "skills": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            "target_audience_text": {"type": ["string", "null"]},
        },
        "required": ["description", "relevant_roles", "industries", "skills", "target_audience_text"],
        "additionalProperties": False,
    }


def build_duplicate_event_merge_prompt(*, fresh: dict[str, Any], previous: list[dict[str, Any]]) -> str:
    return f"""
Самая свежая версия события:
{json.dumps(fresh, ensure_ascii=False, separators=(",", ":"), default=str)}

Предыдущие версии того же события:
{json.dumps(previous, ensure_ascii=False, separators=(",", ":"), default=str)}
""".strip()


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
Сохрани все недатовые поля предыдущего ответа без изменений. Исправляй только start_at, end_at и is_event, если это необходимо из-за некорректных дат. end_at не может быть раньше start_at.
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
