# Prompt: VK Gold Labeling

Use this prompt to create a gold dataset for comparing event extraction results.

```text
Ты размечаешь gold dataset для backend-агента извлечения мероприятий из VK-постов.

Вход:
- JSON selected_posts, где каждый элемент содержит selection_tags, selection_note и post.
- post содержит text/raw_text, source_name, source_url, published_at, external_id.

Задача:
Для каждого post создай ожидаемый результат извлечения.

Статусы:
- extracted: пост является прямым анонсом будущего или актуального события/активности.
- skipped: пост не должен давать event.
- invalid: пост содержит потенциальное событие, но по тексту невозможно определить минимально нужные дату/смысл/объект даже для одного event.

Правила:
- Не выдумывай фактические поля. Если факта нет в тексте или метаданных, ставь null.
- День и месяц без года разрешай через published_at. Если дата уже прошла относительно published_at, используй следующий год.
- Если дата есть, но время не указано, start_at должен быть YYYY-MM-DDT00:00:00.
- start_at/end_at описывают дату и время извлекаемой активности. Для обычного мероприятия это дата проведения, а не дедлайн заявки, не время розыгрыша и не дата итогов.
- Для конкурсов, приемов заявок, голосований и регистраций без отдельной даты проведения event — сама активность участия. start_at — начало активности только если оно явно указано. Если указан только дедлайн/конец, start_at должен быть null, end_at — дата дедлайна/конца.
- Если есть явный период активности "с X по Y", используй X как start_at, Y как end_at.
- Дедлайн, окончание приема, финал, розыгрыш или объявление результатов ставь в end_at только если размечаемая активность — прием заявок/регистрация/голосование/конкурсный период. Не ставь время розыгрыша или объявления победителей как end_at обычного мероприятия.
- Отчеты, итоги, фото/видео-рекапы, поздравления победителей и новости не являются анонсами.
- Пост с отменой/переносом извлекай только если можно идентифицировать конкретное событие и дату.
- Если пост содержит несколько самостоятельных событий с разным временем или смыслом, верни их в events отдельными объектами, а event сделай равным первому объекту из events.
- Если одно мероприятие повторяется в несколько непоследовательных дат с одинаковым временем ("9 и 11 мая в 18:00"), верни отдельный event на каждую дату в events.
- Если это непрерывный диапазон или последовательные дни одной программы ("3-5 января", "3, 4, 5 января"), можно вернуть один event с start_at первой даты и end_at последней даты.
- Дубли группируй через duplicate_group_id. В группе оставляй duplicate_keep=true только у самого актуального анонса, остальные skipped с skip_reason="duplicate_event".

Допустимые skip_reason:
- not_event_announcement
- past_event_report
- duplicate_event
- vacancy
- safety_instruction
- general_news

Верни JSON:
{
  "items": [
    {
      "external_id": "vk:wall...",
      "source_url": "https://vk.com/wall...",
      "expected_status": "extracted|skipped|invalid",
      "skip_reason": null,
      "duplicate_group_id": null,
      "duplicate_keep": true,
      "event": {
        "title": "...",
        "description": null,
        "start_at": "YYYY-MM-DDTHH:MM:SS",
        "end_at": null,
        "timezone": "Europe/Moscow",
        "city": "Волгоград",
        "venue_name": null,
        "address": null,
        "event_type": "EducationEvent",
        "attendance_type": "OfflineEventAttendanceMode",
        "event_status": "EventScheduled",
        "language": "ru",
        "source_name": "...",
        "source_url": "...",
        "raw_text": "...",
        "relevant_roles": ["Participant"],
        "industries": null,
        "skills": null,
        "price_text": "free",
        "target_audience_text": null
      },
      "events": [
        {
          "...": "для extracted: список всех event-объектов; первый объект должен совпадать с event"
        }
      ],
      "notes": "короткое объяснение спорного решения"
    }
  ]
}

Для skipped и invalid event должен быть null, events должен быть null или отсутствовать.
Для extracted skip_reason должен быть null.
Для extracted с одним событием events может отсутствовать или содержать один объект, равный event.
```
