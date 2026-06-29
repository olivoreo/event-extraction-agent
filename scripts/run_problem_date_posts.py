from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from event_extraction_agent import BatchExtractionSettings, ExtractionAgentConfig, ExtractionPipeline, SourcePost  # noqa: E402
from run_golden_extraction import (  # noqa: E402
    DEFAULT_GOLD_PATH,
    DEFAULT_SOURCE_PATH,
    LoggingClient,
    _bool,
    _build_client,
    _build_refinement_client,
    _load_env,
    _log,
    _wrap_refinement_client,
)

DEFAULT_OUTPUT_PATH = ROOT / "var" / "problem_date_posts_result.json"
DEFAULT_PROBLEM_IDS = (
    "vk:wall-45883617_22196",
    "vk:wall-45883617_21986",
    "vk:wall-45883617_21451",
    "vk:wall-45883617_21375",
    "vk:wall-45883617_20068",
    "vk:wall-45883617_19588",
    "vk:wall-45883617_19535",
)


def main() -> None:
    env_path = Path(os.environ.get("ENV_FILE", ROOT / ".sandbox" / ".env"))
    _log(f"loading env: {env_path}")
    env = _load_env(env_path)
    source_path = Path(env.get("PROBLEM_DATE_SOURCE_PATH", env.get("GOLDEN_SOURCE_PATH", DEFAULT_SOURCE_PATH)))
    gold_path = Path(env.get("PROBLEM_DATE_GOLD_PATH", env.get("GOLDEN_GOLD_PATH", DEFAULT_GOLD_PATH)))
    output_path = Path(env.get("PROBLEM_DATE_OUTPUT_PATH", DEFAULT_OUTPUT_PATH))
    post_ids = _post_ids(env.get("PROBLEM_DATE_POST_IDS"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    use_event_type_refinement = _bool(env.get("PROBLEM_DATE_USE_EVENT_TYPE_REFINEMENT", "false"))

    _log(f"problem posts: {', '.join(post_ids)}")
    _log(f"source: {source_path}")
    _log(f"gold: {gold_path}")
    _log(f"output: {output_path}")

    result = ExtractionPipeline(
        source=FilteredStaticJsonSource(source_path, post_ids),
        agent_config=ExtractionAgentConfig(
            main_client=LoggingClient(_build_client(env), "main"),
            refinement_client=_wrap_refinement_client(_build_refinement_client(env)) if use_event_type_refinement else None,
            current_datetime=env.get("CURRENT_DATETIME") or None,
            use_event_type_refinement=use_event_type_refinement,
            min_request_interval_seconds=float(env.get("MIN_REQUEST_INTERVAL_SECONDS", "0")),
            max_retries=int(env.get("MAX_RETRIES", "0")),
        ),
        batch_settings=BatchExtractionSettings(
            skip_duplicates=False,
            skip_event_duplicates=False,
        ),
        save_result_path=output_path,
    ).run()

    _log(f"saved {result.total} outcomes to {output_path}")
    _print_date_comparison(result.model_dump(mode="json").get("outcomes", []), gold_path)


class FilteredStaticJsonSource:
    def __init__(self, path: Path, post_ids: tuple[str, ...]) -> None:
        self.path = path
        self.post_ids = post_ids

    def fetch_posts(self) -> list[SourcePost]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        posts_by_id = {item.get("external_id"): item for item in payload if isinstance(item, dict)}
        missing = [post_id for post_id in self.post_ids if post_id not in posts_by_id]
        if missing:
            raise ValueError(f"missing posts in source: {', '.join(missing)}")
        posts = [SourcePost.model_validate(posts_by_id[post_id]) for post_id in self.post_ids]
        _log(f"loaded {len(posts)} problem posts")
        _log("starting extraction")
        return posts


def _post_ids(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_PROBLEM_IDS
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _print_date_comparison(outcomes: list[dict], gold_path: Path) -> None:
    gold = {
        item["external_id"]: item
        for item in json.loads(gold_path.read_text(encoding="utf-8")).get("items", [])
        if isinstance(item, dict)
    }
    _log("date comparison")
    for outcome in outcomes:
        post_id = outcome.get("post", {}).get("external_id")
        actual_events = outcome.get("events") or ([outcome["event"]] if outcome.get("event") else [])
        gold_item = gold.get(post_id, {})
        expected_events = gold_item.get("events") or ([gold_item["event"]] if gold_item.get("event") else [])
        actual_dates = [(event.get("start_at"), event.get("end_at")) for event in actual_events]
        expected_dates = [(event.get("start_at"), event.get("end_at")) for event in expected_events]
        marker = "OK" if actual_dates == expected_dates else "DIFF"
        print(f"{marker} {post_id} status={outcome.get('status')}", flush=True)
        print(f"  expected: {expected_dates}", flush=True)
        print(f"  actual:   {actual_dates}", flush=True)


if __name__ == "__main__":
    main()
