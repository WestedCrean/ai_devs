from types import SimpleNamespace

from src.ai_devs_core.memory import MemoryObservation, ObservedMemory, ObservationBatch
from src.ai_devs_core.session import RefinedSessionManager


class FakeAgent:
    """Test double for agent-backed memory compression."""

    model_id = "mistral-small-latest"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        """Return a deterministic structured observation response."""
        self.calls.append(kwargs)
        parsed = ObservationBatch(
            observations=[
                MemoryObservation(
                    observation_date="2026-04-27",
                    referenced_date="2026-02-26",
                    relative_date="yesterday",
                    time="06:04",
                    priority="HIGH",
                    content="ECCS8 runaway outlet temperature triggered reactor trip.",
                )
            ]
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))]
        )


def test_observed_memory_instances_do_not_share_state() -> None:
    """ObservedMemory instances should own independent observation lists."""
    first = ObservedMemory()
    second = ObservedMemory()

    first.add("User prefers concise answers.")

    assert first.observations == ["User prefers concise answers."]
    assert second.observations == []


def test_observe_messages_adds_log_like_observation() -> None:
    """Raw messages should be converted into dated log-like observations."""
    memory = ObservedMemory(current_task="Analyze failure logs.")
    agent = FakeAgent()

    memory.observe_messages(
        agent,
        [{"role": "user", "content": "Yesterday ECCS8 failed at 06:04."}],
        observation_date="2026-04-27",
    )

    assert len(memory.observations) == 1
    assert "[HIGH]" in memory.observations[0]
    assert "ECCS8" in memory.observations[0]
    assert agent.calls[0]["response_schema"] is ObservationBatch


def test_refined_session_manager_compresses_into_observed_memory() -> None:
    """RefinedSessionManager should replace old messages with observational memory."""
    agent = FakeAgent()
    manager = RefinedSessionManager(
        agent=agent,
        system_prompt="System prompt.",
        max_context_tokens=10,
        recent_message_count=2,
    )
    manager.messages.extend(
        [
            {"role": "user", "content": "Download logs."},
            {"role": "assistant", "content": "Downloaded failure.csv."},
            {"role": "user", "content": "Verify ECCS8 around 06:04 yesterday."},
            {"role": "assistant", "content": "ECCS8 triggered reactor trip."},
        ]
    )

    manager.compress()

    assert len(manager.messages) == 4
    assert manager.messages[0]["role"] == "system"
    assert str(manager.messages[1]["content"]).startswith("<observational_memory>")
    assert "ECCS8" in str(manager.messages[1]["content"])
    assert manager.messages[-2]["content"] == "Verify ECCS8 around 06:04 yesterday."
    assert manager.messages[-1]["content"] == "ECCS8 triggered reactor trip."
