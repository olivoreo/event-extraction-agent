from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from event_extraction_agent import (  # noqa: E402
    BatchExtractionSettings,
    ExtractionAgentConfig,
    ExtractionPipeline,
    GroqChatClient,
    OllamaChatClient,
    SourcePost,
)


DEFAULT_SOURCE_PATH = ROOT / "tests" / "golden" / "vk_posts_eval_source.json"
DEFAULT_GOLD_PATH = ROOT / "tests" / "golden" / "vk_posts_eval_gold.json"
DEFAULT_OUTPUT_PATH = ROOT / "var" / "golden_eval_result.json"


def main() -> None:
    env_path = Path(os.environ.get("ENV_FILE", ROOT / ".sandbox" / ".env"))
    _log(f"loading env: {env_path}")
    env = _load_env(env_path)
    source_path = Path(env.get("GOLDEN_SOURCE_PATH", DEFAULT_SOURCE_PATH))
    gold_path = Path(env.get("GOLDEN_GOLD_PATH", DEFAULT_GOLD_PATH))
    output_path = Path(env.get("GOLDEN_OUTPUT_PATH", DEFAULT_OUTPUT_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    post_count = _source_post_count(source_path)

    _log(f"source: {source_path} ({post_count} posts)")
    if gold_path.exists():
        _log(f"gold: {gold_path} expected={_gold_status_summary(gold_path)}")
    _log(f"output: {output_path}")
    _log("building LLM clients")

    result = ExtractionPipeline(
        source=StaticJsonSource(source_path),
        agent_config=ExtractionAgentConfig(
            main_client=LoggingClient(_build_client(env), "main"),
            refinement_client=_wrap_refinement_client(_build_refinement_client(env)),
            current_datetime=env.get("CURRENT_DATETIME") or None,
            use_event_type_refinement=_bool(env.get("USE_EVENT_TYPE_REFINEMENT", "true")),
            min_request_interval_seconds=float(env.get("MIN_REQUEST_INTERVAL_SECONDS", "0")),
            max_retries=int(env.get("MAX_RETRIES", "0")),
        ),
        batch_settings=BatchExtractionSettings(
            skip_duplicates=_bool(env.get("SKIP_DUPLICATES", "true")),
            skip_event_duplicates=_bool(env.get("SKIP_EVENT_DUPLICATES", "true")),
        ),
        previous_result_path=str(output_path) if _bool(env.get("INCREMENTAL", "false")) and output_path.exists() else None,
        save_result_path=output_path,
    ).run()

    _log(f"saved {result.total} outcomes to {output_path}")
    _log(
        f"actual=extracted:{result.extracted} skipped:{result.skipped} "
        f"invalid:{result.invalid} llm_errors:{result.llm_errors} cached:{result.cached}"
    )


class StaticJsonSource:
    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch_posts(self) -> list[SourcePost]:
        _log("loading source posts")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("golden source must be a JSON list of SourcePost objects")
        posts = [SourcePost.model_validate(item) for item in payload]
        _log(f"loaded {len(posts)} source posts")
        _log("starting extraction")
        return posts


class LoggingClient:
    def __init__(self, client: OllamaChatClient | GroqChatClient, label: str) -> None:
        self.client = client
        self.label = label
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        model = getattr(self.client, "model", "unknown")
        _log(f"LLM {self.label} request #{self.calls} model={model}")
        response = self.client.complete(system_prompt, user_prompt)
        _log(f"LLM {self.label} request #{self.calls} done")
        return response


def _build_refinement_client(env: dict[str, str]) -> OllamaChatClient | GroqChatClient | None:
    if not env.get("REFINEMENT_LLM_PROVIDER"):
        return None
    return _build_client(env, prefix="REFINEMENT_")


def _wrap_refinement_client(client: OllamaChatClient | GroqChatClient | None) -> LoggingClient | None:
    if client is None:
        _log("refinement client: disabled")
        return None
    _log("refinement client: enabled")
    return LoggingClient(client, "refinement")


def _build_client(env: dict[str, str], prefix: str = "") -> OllamaChatClient | GroqChatClient:
    provider = env.get(f"{prefix}LLM_PROVIDER", "ollama").lower()
    timeout_seconds = float(env.get(f"{prefix}REQUEST_TIMEOUT_SECONDS", env.get("REQUEST_TIMEOUT_SECONDS", "120")))

    if provider == "groq":
        return GroqChatClient(
            api_key=_required(env, f"{prefix}GROQ_API_KEY", fallback_key="GROQ_API_KEY"),
            model=env.get(f"{prefix}GROQ_MODEL", env.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")),
            timeout_seconds=timeout_seconds,
            max_retries=int(env.get(f"{prefix}GROQ_MAX_RETRIES", env.get("GROQ_MAX_RETRIES", "3"))),
        )
    if provider != "ollama":
        raise ValueError("LLM_PROVIDER must be ollama or groq")
    return OllamaChatClient(
        model=env.get(f"{prefix}OLLAMA_MODEL", env.get("OLLAMA_MODEL", "qwen2.5:3b")),
        host=env.get(f"{prefix}OLLAMA_HOST", env.get("OLLAMA_HOST", "http://localhost:11434")),
        timeout_seconds=timeout_seconds,
    )


def _gold_status_summary(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(item["expected_status"] for item in data.get("items", []))
    return " ".join(f"{key}:{counts[key]}" for key in ("extracted", "skipped", "invalid"))


def _source_post_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("golden source must be a JSON list of SourcePost objects")
    return len(payload)


def _load_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if not path.exists():
        return env

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip("'\""))
    return env


def _required(env: dict[str, str], key: str, fallback_key: str | None = None) -> str:
    value = env.get(key, "").strip()
    if not value and fallback_key is not None:
        value = env.get(fallback_key, "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _log(message: str) -> None:
    print(f"[golden] {message}", flush=True)


if __name__ == "__main__":
    main()
