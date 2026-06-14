# event_extraction_agent

`event_extraction_agent` - небольшая Python-библиотека для извлечения структурированных данных о мероприятиях из текстовых постов с помощью LLM.

Версия `0.4.0` решает задачу extraction через единый стабильный entrypoint `ExtractionPipeline`: вы передаете источник `SourcePost`, `ExtractionAgentConfig` и получаете `BatchExtractionResult`. Pipeline отвечает за batch extraction, incremental processing и сохранение результата между запусками. Также есть `SourceAdapter` для собственных источников и production-адаптер `VKSource`. Хранение в базе данных, HTTP API и расписания намеренно не входят в пакет.

## Установка

Локально из этого репозитория:

```bash
python -m pip install -e .
```

Для разработки:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Быстрый старт

```python
from event_extraction_agent import ExtractionAgentConfig, ExtractionPipeline, OllamaChatClient, SourcePost

class MySource:
    def fetch_posts(self) -> list[SourcePost]:
        return [
            SourcePost(
                text="12 июня в 18:00 пройдет лекция.",
                source_name="My source",
                external_id="post-1",
            )
        ]


pipeline = ExtractionPipeline(
    source=MySource(),
    agent_config=ExtractionAgentConfig(
        main_client=OllamaChatClient(model="qwen2.5:3b"),
    ),
    save_result_path="events_result.json",
)

result = pipeline.run()
print(result.extracted, result.skipped, result.invalid, result.llm_errors)
```

`ExtractionPipeline` - основной стабильный entrypoint библиотеки. Пользователь передает источник постов, `ExtractionAgentConfig` и настройки запуска, а получает `BatchExtractionResult` со всеми `ExtractionOutcome`, счетчиками, ошибками и persistence-методами.

## Incremental processing

Pipeline умеет сам загрузить предыдущий результат и сохранить новый:

```python
pipeline = ExtractionPipeline(
    source=source,
    agent_config=agent_config,
    previous_result_path="events_result.json",
    save_result_path="events_result.json",
)

result = pipeline.run()
```

Incremental-режим пропускает LLM extraction, если у нового поста совпали `external_id` и нормализованный текст для LLM (`raw_text`, если он задан, иначе `text`) с предыдущим `ExtractionOutcome`. Если предыдущий результат был `llm_error`, пост по умолчанию обрабатывается повторно.

Те же параметры можно передать на один конкретный запуск:

```python
result = pipeline.run(
    previous_result_path="events_result.json",
    save_result_path="events_result.json",
)
```

## Source adapters

`SourceAdapter` отвечает только за получение подготовленных `SourcePost`:

```python
class MySource:
    def fetch_posts(self) -> list[SourcePost]:
        return [
            SourcePost(
                text="12 июня в 18:00 пройдет лекция.",
                source_name="My source",
                external_id="post-1",
            )
        ]
```

Batch-обработка внутри pipeline последовательная и сохраняет порядок входных постов. По умолчанию pipeline пропускает дубли по `external_id`, а если `external_id` нет - по нормализованному тексту. `max_errors` в `BatchExtractionSettings` ограничивает число `invalid` и `llm_error`; после достижения лимита оставшиеся посты возвращаются как `skipped` с ошибкой `error_limit_reached`.

## VK source

```python
from event_extraction_agent import ExtractionAgentConfig, ExtractionPipeline, OllamaChatClient, VKSource

source = VKSource(
    access_token="vk-service-token",
    sources=[
        "https://vk.com/club123",
        "public456",
        "my_community_domain",
        -789,
    ],
    posts_per_source_limit=20,
)

result = ExtractionPipeline(
    source=source,
    agent_config=ExtractionAgentConfig(
        main_client=OllamaChatClient(model="qwen2.5:3b"),
    ),
    previous_result_path="events_result.json",
    save_result_path="events_result.json",
).run()
```

`VKSource` принимает токен строкой и не читает `.env`. Источники можно задавать как URL, `club...`, `public...`, домены или числовые `owner_id`. Адаптер получает посты через `wall.get`, сохраняет исходный текст поста в `SourcePost.text`, очищенный текст для LLM в `SourcePost.raw_text`, добавляет полезные метаданные (`source_name`, `source_url`, `published_at`, `external_id`) и приводит все к `SourcePost`.

Если один VK source временно или постоянно недоступен, `VKSource` по умолчанию продолжает обрабатывать остальные источники. Ошибки доступны после вызова через `source.errors` или вместе с постами через `fetch_posts_with_errors()`:

```python
fetch_result = source.fetch_posts_with_errors()

posts = fetch_result.posts
for error in fetch_result.errors:
    print(error.source, error.code, error)
```

Для fail-fast поведения можно передать `continue_on_source_error=False`. `VKApiError` содержит `code`, `details`, `method`, `source` и `retryable`, поэтому прод-код не теряет информацию, какой именно source упал.

Для временных ошибок `VKSource` делает retry/backoff. По умолчанию `max_retries=3`, `retry_backoff_seconds=1.0`, а `rate_limit_per_second=20`, что соответствует лимиту 20 запросов в секунду для сервисного API VK. Retry применяется к HTTP `429`/`5xx`, сетевым ошибкам и временным VK-кодам вроде `6`, `9`, `10`, `29`.

Вложения не передаются в агент. Если VK-пост не содержит текста после очистки, адаптер его пропускает. `raw_text` очищается от emoji, переносов строк и лишних пробелов.

## Настройка моделей и уточнений

```python
from event_extraction_agent import (
    ExtractionAgentConfig,
    ExtractionPipeline,
    OllamaChatClient,
    SourcePost,
)

main_client = OllamaChatClient(model="qwen2.5:3b")
refinement_client = OllamaChatClient(model="qwen2.5:7b")

config = ExtractionAgentConfig(
    main_client=main_client,
    refinement_client=refinement_client,
    use_event_type_refinement=True,
    current_datetime="2026-06-10T12:00:00+03:00",
    request_timeout_seconds=120,
    min_request_interval_seconds=2.1,
    max_retries=1,
)

class OnePostSource:
    def fetch_posts(self) -> list[SourcePost]:
        return [SourcePost(text="12 июня в 18:00 пройдет лекция.")]


result = ExtractionPipeline(source=OnePostSource(), agent_config=config).run()
outcome = result.outcomes[0]

print(outcome.raw_llm_metadata)
```

`current_datetime` передается в prompt и используется моделью для оценки актуальности и статуса мероприятия. Это поведение перенесено из исходного `event-ai-agent`, но оформлено как настройка библиотеки.

`refinement_client` - вспомогательная модель для уточнений. Сейчас библиотека использует ее только для второго прохода по `event_type`, если основной extraction вернул отсутствующий, недопустимый или слишком общий тип события (`SocialEvent`). В будущем этот же клиент можно использовать для уточнения других подозрительных полей без изменения пользовательского API.

Все настройки агента передаются через `ExtractionAgentConfig`. Это единственный стабильный способ настроить LLM-клиентов и поведение extraction внутри pipeline.

Библиотека не ограничивает список нейросетей. Любой объект с методом `complete(system_prompt, user_prompt) -> str` может быть клиентом:

```python
class MyLLMClient:
    model = "my-provider/my-model"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # Вызов вашего runtime, API, локальной модели или gateway.
        return '{"is_event": false, "skip_reason": "not_event_announcement", "event": null}'


pipeline = ExtractionPipeline(
    source=OnePostSource(),
    agent_config=ExtractionAgentConfig(
        main_client=MyLLMClient(),
        current_datetime="2026-06-10T12:00:00+03:00",
    ),
)
```

## Groq

```python
from event_extraction_agent import ExtractionAgentConfig, ExtractionPipeline, GroqChatClient, SourcePost

client = GroqChatClient(
    api_key="gsk_...",
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

class OnePostSource:
    def fetch_posts(self) -> list[SourcePost]:
        return [
            SourcePost(
                text="20 июля в 15:30 пройдет открытый мастер-класс для начинающих дизайнеров.",
                source_name="Example source",
                source_url="https://example.com/posts/456",
                published_at="2026-07-10T12:00:00+03:00",
                external_id="post-456",
            )
        ]


result = ExtractionPipeline(
    source=OnePostSource(),
    agent_config=ExtractionAgentConfig(main_client=client),
).run()
```

## Публичные модели

`SourcePost` - входная модель поста:

- `text`
- `raw_text`
- `source_name`
- `source_url`
- `published_at`
- `external_id`

Обязательное поле только `text`. `raw_text` опционален и используется как очищенный текст для LLM. Если `raw_text` не задан, агент использует `text`. Остальные поля опциональны, но полезны для трассировки результата и восстановления даты события, если в тексте указан день и месяц без года.

`ExtractionOutcome` - результат работы агента:

- `status`: `extracted`, `skipped`, `invalid` или `llm_error`
- `event`: модель `Event`, если извлечение успешно
- `post`: исходный `SourcePost`
- `errors`: структурированные ошибки
- `raw_llm_metadata`: отладочные метаданные по моделям, текущей дате prompt, refinement-проходам и LLM-попыткам

`ExtractionAgentConfig` - настройки агента:

- `main_model`
- `refinement_model`
- `main_client`
- `refinement_client`
- `use_event_type_refinement`
- `current_datetime`
- `request_timeout_seconds`
- `min_request_interval_seconds`
- `max_retries`

`SourceAdapter` - протокол источника:

- `fetch_posts() -> list[SourcePost]`

`ExtractionPipeline` - orchestration-слой:

- получает посты из `SourceAdapter`
- запускает batch или incremental extraction
- при необходимости загружает предыдущий `BatchExtractionResult`
- при необходимости сохраняет новый `BatchExtractionResult`

`VKSource` - готовый source adapter для VK:

- `access_token`
- `sources`
- `posts_per_source_limit`
- `offset`
- `wall_filter`
- `api_version`
- `timeout_seconds`
- `batch_size`
- `continue_on_source_error`
- `rate_limit_per_second`
- `max_retries`
- `retry_backoff_seconds`

`BatchExtractionSettings` - настройки batch-обработки:

- `mode`: сейчас только `sequential`
- `max_errors`: опциональный лимит ошибок `invalid` и `llm_error`
- `skip_empty`: пропуск пустых текстов, если такие попадут в пачку
- `skip_duplicates`: пропуск дублирующихся постов

`BatchExtractionResult` - сводный результат batch-обработки:

- `settings`
- `outcomes`
- `total`
- `extracted`
- `skipped`
- `invalid`
- `llm_errors`
- `cached`
- `processed`
- `error_count`
- `error_limit_reached`

Дополнительно `BatchExtractionResult.save_json(path)` сохраняет результат в UTF-8 JSON, а `BatchExtractionResult.load_json(path)` загружает его обратно для повторного использования, например в `existing_outcomes` incremental pipeline.

`Event` содержит поля события: название, описание, даты, таймзону, место, тип события, формат участия, статус, язык, метаданные источника, исходный текст, роли, индустрии, навыки, стоимость и текст целевой аудитории.

## Текущий scope

Входит в v0.4.0:

- переиспользуемый Python-пакет
- типизированные входные и выходные модели
- Ollama-compatible chat client
- Groq OpenAI-compatible chat client
- промпты, валидация, легкое исправление ответа модели и уточнение типа события
- последовательная batch-обработка подготовленных `SourcePost`
- лимит ошибок, сохранение порядка, пропуск дублей и явная batch-сводка
- incremental processing по `external_id` и нормализованному тексту для LLM
- persistence для `BatchExtractionResult` через `save_json` и `load_json`
- `ExtractionAgentConfig` для настройки моделей, времени prompt, timeout, rate limit и retries
- вспомогательная модель уточнений для подозрительных полей; сейчас используется для `event_type`
- прозрачные `raw_llm_metadata` с LLM-попытками и выбранными моделями
- `SourceAdapter` как протокол источников
- `ExtractionPipeline` как единый стабильный entrypoint запуска
- production-адаптер `VKSource` для текстовых постов VK
- partial failure handling, retry/backoff и rate limit для `VKSource`

Не входит в v0.4.0:

- отправка VK-вложений в агент
- чтение токенов или источников из `.env`
- адаптеры внешних источников, кроме VK
- разбор вложений
- хранение в базе данных
- FastAPI/backend routes
- scheduled jobs
- custom source-specific ingestion за пределами готового `VKSource`
- локальные ML-классификаторы

Приложение, которое использует этот пакет, отвечает за конфигурацию, секреты, хранение данных и свои дополнительные source adapters.

## Происхождение пакета

`event_extraction_agent` выделен из проекта `event-ai-agent`, который использовался как основа для первой версии библиотеки. Из исходного проекта перенесено только extraction-ядро: модели события, промпты, валидация, легкое исправление ответа LLM и клиенты для Ollama/Groq.

При переносе намеренно не включались backend API, база данных, загрузчики внешних источников и экспериментальные ML-компоненты. Цель пакета - сделать extraction-логику переиспользуемой в других проектах.
