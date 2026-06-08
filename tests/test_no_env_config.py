from pathlib import Path


def test_runtime_code_does_not_read_os_environment():
    source_root = Path(__file__).resolve().parents[1] / "src" / "event_extraction_agent"
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))

    assert "os.getenv" not in source
    assert "import os" not in source
