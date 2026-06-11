from __future__ import annotations

from dataclasses import dataclass

from event_extraction_agent.agent import ExtractionAgent
from event_extraction_agent.models import BatchExtractionResult, BatchExtractionSettings, SourcePost
from event_extraction_agent.sources import SourceAdapter


@dataclass(frozen=True)
class ExtractionPipeline:
    """Orchestrate source fetching and batch extraction."""

    agent: ExtractionAgent
    source: SourceAdapter
    batch_settings: BatchExtractionSettings | None = None

    def run(self) -> BatchExtractionResult:
        posts = self.source.fetch_posts()
        _validate_posts(posts)
        return self.agent.extract_batch(posts, settings=self.batch_settings)


def extract_from_source(
    source: SourceAdapter,
    agent: ExtractionAgent,
    batch_settings: BatchExtractionSettings | None = None,
) -> BatchExtractionResult:
    """Fetch prepared posts from a source and extract events from them."""

    return ExtractionPipeline(agent=agent, source=source, batch_settings=batch_settings).run()


def _validate_posts(posts: object) -> None:
    if not isinstance(posts, list):
        raise TypeError("source.fetch_posts() must return list[SourcePost]")
    if not all(isinstance(post, SourcePost) for post in posts):
        raise TypeError("source.fetch_posts() must return list[SourcePost]")
