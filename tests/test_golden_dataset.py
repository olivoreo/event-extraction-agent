import json
from pathlib import Path

from event_extraction_agent import SourcePost


GOLDEN_DIR = Path(__file__).parent / "golden"


def test_golden_eval_source_matches_gold_items():
    source = json.loads((GOLDEN_DIR / "vk_posts_eval_source.json").read_text(encoding="utf-8"))
    gold = json.loads((GOLDEN_DIR / "vk_posts_eval_gold.json").read_text(encoding="utf-8"))["items"]

    posts = [SourcePost.model_validate(item) for item in source]
    source_ids = [post.external_id for post in posts]
    gold_ids = [item["external_id"] for item in gold]

    assert len(posts) == 76
    assert len(gold) == 76
    assert source_ids == gold_ids


def test_golden_eval_gold_shape_is_consistent():
    gold = json.loads((GOLDEN_DIR / "vk_posts_eval_gold.json").read_text(encoding="utf-8"))["items"]

    for item in gold:
        status = item["expected_status"]
        assert status in {"extracted", "skipped", "invalid"}
        if status == "extracted":
            assert item["event"] is not None
            assert item["skip_reason"] is None
            assert item["duplicate_keep"] is True
            if item.get("events") is not None:
                assert item["events"]
                assert item["events"][0] == item["event"]
        if status == "skipped":
            assert item["event"] is None
            assert item["skip_reason"]
        if status == "invalid":
            assert item["event"] is None
            assert item["skip_reason"] is None
