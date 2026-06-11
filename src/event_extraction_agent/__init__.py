"""Public API for the event extraction package."""

from event_extraction_agent.agent import (
    ExtractionAgent,
    GroqChatClient,
    LLMClient,
    OllamaChatClient,
    RequestRateLimiter,
)
from event_extraction_agent.models import (
    AttendanceType,
    BatchExtractionResult,
    BatchExtractionSettings,
    BatchMode,
    Event,
    EventStatus,
    EventType,
    ExtractionAgentConfig,
    ExtractionError,
    ExtractionOutcome,
    ExtractionStatus,
    SourcePost,
)
from event_extraction_agent.pipeline import ExtractionPipeline, extract_from_source
from event_extraction_agent.sources import SourceAdapter
from event_extraction_agent.vk import (
    VKApiClient,
    VKApiError,
    VKPostSource,
    VKSource,
    build_vk_post_url,
    parse_vk_source,
)

__all__ = [
    "AttendanceType",
    "BatchExtractionResult",
    "BatchExtractionSettings",
    "BatchMode",
    "Event",
    "EventStatus",
    "EventType",
    "ExtractionAgentConfig",
    "ExtractionAgent",
    "ExtractionError",
    "ExtractionOutcome",
    "ExtractionStatus",
    "ExtractionPipeline",
    "extract_from_source",
    "GroqChatClient",
    "LLMClient",
    "OllamaChatClient",
    "RequestRateLimiter",
    "SourceAdapter",
    "SourcePost",
    "VKApiClient",
    "VKApiError",
    "VKPostSource",
    "VKSource",
    "build_vk_post_url",
    "parse_vk_source",
]
