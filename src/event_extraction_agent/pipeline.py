from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from event_extraction_agent.agent import ExtractionAgent, _batch_result
from event_extraction_agent.models import (
    BatchExtractionResult,
    BatchExtractionSettings,
    ExtractionAgentConfig,
    ExtractionOutcome,
    SourcePost,
)
from event_extraction_agent.sources import SourceAdapter


@dataclass(frozen=True, init=False)
class ExtractionPipeline:
    """Stable entrypoint for fetching posts and extracting events."""

    source: SourceAdapter
    agent: ExtractionAgent
    batch_settings: BatchExtractionSettings | None = None
    existing_outcomes: list[ExtractionOutcome] | None = None
    previous_result_path: str | Path | None = None
    save_result_path: str | Path | None = None
    retry_cached_llm_errors: bool = True
    accumulate_existing_outcomes: bool = False

    def __init__(
        self,
        source: SourceAdapter,
        *,
        agent_config: ExtractionAgentConfig,
        batch_settings: BatchExtractionSettings | None = None,
        existing_outcomes: list[ExtractionOutcome] | None = None,
        previous_result_path: str | Path | None = None,
        save_result_path: str | Path | None = None,
        retry_cached_llm_errors: bool = True,
        accumulate_existing_outcomes: bool = False,
    ) -> None:
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "agent", ExtractionAgent(config=agent_config))
        object.__setattr__(self, "batch_settings", batch_settings)
        object.__setattr__(self, "existing_outcomes", existing_outcomes)
        object.__setattr__(self, "previous_result_path", previous_result_path)
        object.__setattr__(self, "save_result_path", save_result_path)
        object.__setattr__(self, "retry_cached_llm_errors", retry_cached_llm_errors)
        object.__setattr__(self, "accumulate_existing_outcomes", accumulate_existing_outcomes)

    def run(
        self,
        *,
        existing_outcomes: list[ExtractionOutcome] | None = None,
        previous_result_path: str | Path | None = None,
        save_result_path: str | Path | None = None,
        accumulate_existing_outcomes: bool | None = None,
    ) -> BatchExtractionResult:
        posts = self.source.fetch_posts()
        _validate_posts(posts)

        resolved_existing_outcomes = _resolve_existing_outcomes(
            explicit_outcomes=existing_outcomes,
            configured_outcomes=self.existing_outcomes,
            explicit_path=previous_result_path,
            configured_path=self.previous_result_path,
        )
        if resolved_existing_outcomes is not None:
            result = self.agent.extract_incremental(
                posts,
                existing_outcomes=resolved_existing_outcomes,
                settings=self.batch_settings,
                retry_llm_errors=self.retry_cached_llm_errors,
            )
            should_accumulate = (
                self.accumulate_existing_outcomes
                if accumulate_existing_outcomes is None
                else accumulate_existing_outcomes
            )
            if should_accumulate:
                result = _batch_result(
                    _accumulate_outcomes(result.outcomes, resolved_existing_outcomes),
                    settings=self.batch_settings,
                    error_limit_reached=result.error_limit_reached,
                )
        else:
            result = self.agent.extract_batch(posts, settings=self.batch_settings)

        result_path = save_result_path if save_result_path is not None else self.save_result_path
        if result_path is not None:
            result.save_json(result_path)
        return result


def _resolve_existing_outcomes(
    *,
    explicit_outcomes: list[ExtractionOutcome] | None,
    configured_outcomes: list[ExtractionOutcome] | None,
    explicit_path: str | Path | None,
    configured_path: str | Path | None,
) -> list[ExtractionOutcome] | None:
    if explicit_outcomes is not None:
        return explicit_outcomes
    if configured_outcomes is not None:
        return configured_outcomes

    path = explicit_path if explicit_path is not None else configured_path
    if path is None:
        return None
    return BatchExtractionResult.load_json(path).outcomes


def _validate_posts(posts: object) -> None:
    if not isinstance(posts, list):
        raise TypeError("source.fetch_posts() must return list[SourcePost]")
    if not all(isinstance(post, SourcePost) for post in posts):
        raise TypeError("source.fetch_posts() must return list[SourcePost]")


def _accumulate_outcomes(
    current_outcomes: list[ExtractionOutcome],
    existing_outcomes: list[ExtractionOutcome],
) -> list[ExtractionOutcome]:
    current_ids = {outcome.post.external_id for outcome in current_outcomes if outcome.post.external_id is not None}
    return current_outcomes + [
        outcome
        for outcome in existing_outcomes
        if outcome.post.external_id is None or outcome.post.external_id not in current_ids
    ]
