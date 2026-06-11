from __future__ import annotations

from dataclasses import dataclass

from event_extraction_agent.agent import ExtractionAgent
from event_extraction_agent.models import BatchExtractionResult, BatchExtractionSettings, ExtractionOutcome, SourcePost
from event_extraction_agent.sources import SourceAdapter


@dataclass(frozen=True)
class ExtractionPipeline:
    """Orchestrate source fetching and batch extraction."""

    agent: ExtractionAgent
    source: SourceAdapter
    batch_settings: BatchExtractionSettings | None = None
    existing_outcomes: list[ExtractionOutcome] | None = None
    retry_cached_llm_errors: bool = True

    def run(self) -> BatchExtractionResult:
        posts = self.source.fetch_posts()
        _validate_posts(posts)
        if self.existing_outcomes is not None:
            return self.agent.extract_incremental(
                posts,
                existing_outcomes=self.existing_outcomes,
                settings=self.batch_settings,
                retry_llm_errors=self.retry_cached_llm_errors,
            )
        return self.agent.extract_batch(posts, settings=self.batch_settings)


def extract_from_source(
    source: SourceAdapter,
    agent: ExtractionAgent,
    batch_settings: BatchExtractionSettings | None = None,
    existing_outcomes: list[ExtractionOutcome] | None = None,
    retry_cached_llm_errors: bool = True,
) -> BatchExtractionResult:
    """Fetch prepared posts from a source and extract events from them."""

    return ExtractionPipeline(
        agent=agent,
        source=source,
        batch_settings=batch_settings,
        existing_outcomes=existing_outcomes,
        retry_cached_llm_errors=retry_cached_llm_errors,
    ).run()


def _validate_posts(posts: object) -> None:
    if not isinstance(posts, list):
        raise TypeError("source.fetch_posts() must return list[SourcePost]")
    if not all(isinstance(post, SourcePost) for post in posts):
        raise TypeError("source.fetch_posts() must return list[SourcePost]")
