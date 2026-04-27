from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

lesson = importlib.import_module("src.lessons.s02e03.main")


@pytest.fixture(autouse=True)
def isolated_lesson_state(tmp_path: Path, monkeypatch) -> None:
    """Run s02e03 tool tests against isolated file and memory state."""
    monkeypatch.setattr(lesson, "MCP_FILES_PATH", tmp_path)
    lesson._reset_lesson_state()
    yield
    lesson._reset_lesson_state()


def test_collect_log_candidates_writes_filtered_file(tmp_path: Path) -> None:
    """Candidate collection should write broad matches to a file and return metadata."""
    source = tmp_path / "failure.csv"
    source.write_text(
        "\n".join(
            [
                "2026-04-26 10:00:00 [INFO] PWR01 routine nominal",
                "2026-04-26 10:01:00 [WARN] PWR01 input ripple crossed limits",
                "2026-04-26 10:02:00 [CRIT] ECCS8 cooling cannot maintain gradient",
                "2026-04-26 10:03:00 [ERRO] PWR01 telemetry missing",
                "2026-04-26 10:04:00 [WARN] AUX01 unrelated warning",
            ]
        ),
        encoding="utf-8",
    )

    result = lesson.collect_log_candidates(
        filename="failure.csv",
        output_filename="pwr_candidates.txt",
        severities="WARN,ERRO,CRIT",
        components="PWR01",
        include_patterns="ripple,telemetry",
        exclude_patterns="",
    )

    output = (tmp_path / "pwr_candidates.txt").read_text(encoding="utf-8").splitlines()
    assert "Written candidates: 2" in result
    assert len(output) == 2
    assert output[0].startswith("failure.csv:2:")
    assert "PWR01 input ripple" in output[0]
    assert "PWR01 telemetry missing" in output[1]


def test_add_failure_observations_from_file_loads_pages_and_dedupes(
    tmp_path: Path,
) -> None:
    """Loading candidate files should page data into memory without duplicates."""
    (tmp_path / "candidates.txt").write_text(
        "\n".join(
            [
                "failure.csv:10:2026-04-26 10:01:00 [WARN] PWR01 input ripple",
                "failure.csv:11:2026-04-26 10:02:00 [CRIT] ECCS8 cooling failed",
                "failure.csv:12:2026-04-26 10:03:00 [ERRO] PWR01 telemetry missing",
            ]
        ),
        encoding="utf-8",
    )

    first = lesson.add_failure_observations_from_file(
        filename="candidates.txt",
        offset=0,
        limit=2,
    )
    second = lesson.add_failure_observations_from_file(
        filename="candidates.txt",
        offset=0,
        limit=2,
    )

    assert "Loaded 2 new observations" in first
    assert "Next offset: 2" in first
    assert "Loaded 0 new observations" in second
    assert "duplicates skipped: 2" in second
    assert len(lesson.failure_memory.observations) == 2


def test_search_missing_component_uses_default_output_file(tmp_path: Path) -> None:
    """Missing component search should collect that component into a file."""
    (tmp_path / "failure.csv").write_text(
        "\n".join(
            [
                "2026-04-26 10:01:00 [WARN] PWR01 input ripple",
                "2026-04-26 10:02:00 [CRIT] ECCS8 cooling failed",
                "2026-04-26 10:03:00 [ERRO] PWR01 telemetry missing",
            ]
        ),
        encoding="utf-8",
    )

    result = lesson.search_missing_component("PWR01")

    assert "missing_PWR01.txt" in result
    output = (tmp_path / "missing_PWR01.txt").read_text(encoding="utf-8")
    assert "PWR01 input ripple" in output
    assert "ECCS8" not in output


def test_verify_failure_logs_rejects_oversized_payload_before_api(monkeypatch) -> None:
    """Oversized logs should not call the verification API."""
    calls = {"verify": 0}

    def fail_verify(task: str, answer: dict) -> dict:
        calls["verify"] += 1
        return {"task": task, "answer": answer}

    monkeypatch.setattr(lesson.ai_devs_core, "verify", fail_verify)

    result = lesson.verify_failure_logs("token " * 5_000)

    assert "above the 1500 token limit" in result
    assert calls["verify"] == 0


def test_verify_failure_logs_stores_feedback(monkeypatch) -> None:
    """Verification feedback should be retained for the next workflow step."""
    response = {"status": 400, "response": {"message": "Missing PWR01"}}
    monkeypatch.setattr(lesson.ai_devs_core, "verify", lambda task, answer: response)

    result = lesson.verify_failure_logs(
        "2026-04-26 10:01 [WARN] PWR01 input ripple"
    )

    assert result == str(response)
    assert lesson.latest_verification_feedback == str(response)
    assert any("Missing PWR01" in item for item in lesson.failure_memory.observations)


def test_create_native_tools_exposes_task_tools_only() -> None:
    """Raw verify should not be exposed as a lesson tool."""
    tools = lesson.create_native_tools()
    names = {tool.__name__ for tool in tools}

    assert {
        "collect_log_candidates",
        "add_failure_observations_from_file",
        "search_missing_component",
        "verify_failure_logs",
    } <= names
    assert "verify" not in names


def test_main_uses_mistral_small_factory(monkeypatch) -> None:
    """The interactive agent should be created through the Mistral small factory path."""
    calls: list[tuple[str, str]] = []

    def fake_create_agent(provider: str, model_id: str) -> SimpleNamespace:
        calls.append((provider, model_id))
        raise KeyboardInterrupt

    monkeypatch.setattr(lesson, "create_agent", fake_create_agent)

    with pytest.raises(KeyboardInterrupt):
        lesson.main()

    assert calls == [("mistral", "mistral-small-latest")]
