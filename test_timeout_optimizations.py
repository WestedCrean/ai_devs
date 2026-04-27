from types import SimpleNamespace

import httpx

from src.ai_devs_core import agent as agent_module
from src.ai_devs_core.agent import (
    FAgent,
    MISTRAL_CHAT_MAX_TOKENS,
    MISTRAL_TIMEOUT_MS,
)
from src.ai_devs_core.job_client import ErrorClassifier
from src.ai_devs_core.session import BaseSessionManager


def test_read_timeout_is_retryable() -> None:
    """ReadTimeout should be treated as retryable."""
    assert ErrorClassifier.is_retryable(
        httpx.ReadTimeout("The read operation timed out")
    )


def test_fagent_completion_uses_timeout_and_max_tokens(monkeypatch) -> None:
    """Normal Mistral chat calls should include timeout and output bounds."""
    calls: dict[str, object] = {}

    class FakeChat:
        def complete(self, **kwargs):
            calls["complete_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="done", tool_calls=None)
                    )
                ]
            )

    class FakeMistral:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr(agent_module, "Mistral", FakeMistral)
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

    agent = FAgent.__new__(FAgent)
    agent.model_id = "mistral-large-latest"
    agent.config = SimpleNamespace(MISTRAL_API_KEY="test-key")

    response = agent.chat_completion(chat_history=[{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "done"
    assert calls["client_kwargs"]["timeout_ms"] == MISTRAL_TIMEOUT_MS
    assert calls["complete_kwargs"]["timeout_ms"] == MISTRAL_TIMEOUT_MS
    assert calls["complete_kwargs"]["max_tokens"] == MISTRAL_CHAT_MAX_TOKENS


def test_fagent_stream_retries_read_timeout(monkeypatch) -> None:
    """Streamed Mistral calls should retry ReadTimeout at the step level."""
    calls = {"stream_count": 0}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __iter__(self):
            delta = SimpleNamespace(content="done", tool_calls=None)
            choice = SimpleNamespace(delta=delta)
            data = SimpleNamespace(choices=[choice])
            return iter([SimpleNamespace(data=data)])

    class FakeChat:
        def stream(self, **kwargs):
            calls["stream_kwargs"] = kwargs
            calls["stream_count"] += 1
            if calls["stream_count"] == 1:
                raise httpx.ReadTimeout("The read operation timed out")
            return FakeStream()

    class FakeMistral:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr(agent_module, "Mistral", FakeMistral)
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

    agent = FAgent.__new__(FAgent)
    agent.model_id = "mistral-large-latest"
    agent.config = SimpleNamespace(MISTRAL_API_KEY="test-key")

    response = agent.chat_completion(
        chat_history=[{"role": "user", "content": "hi"}],
        stream=True,
    )

    assert calls["stream_count"] == 2
    assert calls["stream_kwargs"]["timeout_ms"] == MISTRAL_TIMEOUT_MS
    assert response.choices[0].message.content == "done"


def test_fagent_stream_falls_back_when_tool_arguments_are_incomplete(
    monkeypatch,
) -> None:
    """Incomplete streamed tool arguments should be recovered with complete()."""
    calls = {"stream_count": 0, "complete_count": 0}
    tool_results: list[tuple[str, str]] = []

    def lookup(value: str) -> str:
        """Lookup test values."""
        return f"found:{value}"

    invalid_tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="lookup", arguments='{"value": "abc'),
    )
    valid_tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="lookup", arguments='{"value": "abc"}'),
    )

    class FakeStream:
        def __init__(self, events):
            self.events = events

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __iter__(self):
            return iter(self.events)

    def stream_event(content: str = "", tool_calls: list | None = None):
        delta = SimpleNamespace(content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(delta=delta)
        data = SimpleNamespace(choices=[choice])
        return SimpleNamespace(data=data)

    class FakeChat:
        def stream(self, **kwargs):
            calls["stream_count"] += 1
            if calls["stream_count"] == 1:
                return FakeStream([stream_event(tool_calls=[invalid_tool_call])])
            return FakeStream([stream_event(content="done")])

        def complete(self, **kwargs):
            calls["complete_count"] += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="", tool_calls=[valid_tool_call])
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class FakeMistral:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr(agent_module, "Mistral", FakeMistral)
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

    agent = FAgent.__new__(FAgent)
    agent.model_id = "mistral-large-latest"
    agent.config = SimpleNamespace(MISTRAL_API_KEY="test-key")

    response = agent.chat_completion(
        chat_history=[{"role": "user", "content": "use tool"}],
        tools=[lookup],
        on_tool_result=lambda name, result: tool_results.append((name, result)),
        stream=True,
        max_steps=2,
    )

    assert calls == {"stream_count": 2, "complete_count": 1}
    assert tool_results == [("lookup", "found:abc")]
    assert response.choices[0].message.content == "done"


def test_fagent_non_stream_history_normalizes_structured_content(monkeypatch) -> None:
    """Structured provider content should not be replayed into later requests."""
    calls: list[dict] = []
    structured_content = [
        {
            "thinking": [
                {
                    "text": "I need to inspect more log lines.",
                    "type": "text",
                }
            ],
            "type": "thinking",
            "signature": None,
            "closed": True,
        }
    ]

    class FakeChat:
        def complete(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=structured_content,
                                tool_calls=[
                                    SimpleNamespace(
                                        id="call-1",
                                        function=SimpleNamespace(
                                            name="lookup",
                                            arguments='{"value": "abc"}',
                                        ),
                                    )
                                ],
                            )
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="done", tool_calls=None)
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

    class FakeMistral:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    def lookup(value: str) -> str:
        """Lookup test values."""
        return f"found:{value}"

    monkeypatch.setattr(agent_module, "Mistral", FakeMistral)
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

    agent = FAgent.__new__(FAgent)
    agent.model_id = "mistral-large-latest"
    agent.config = SimpleNamespace(MISTRAL_API_KEY="test-key")

    response = agent.chat_completion(
        chat_history=[{"role": "user", "content": "use tool"}],
        tools=[lookup],
        max_steps=2,
    )

    assert response.choices[0].message.content == "done"
    assert calls[1]["messages"][1]["content"] == "I need to inspect more log lines."


def test_session_summarize_uses_timeout_and_max_tokens(monkeypatch) -> None:
    """Session compression summarization should use bounded Mistral requests."""
    calls: dict[str, object] = {}

    class FakeChat:
        def parse(self, **kwargs):
            calls["parse_kwargs"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            parsed=SimpleNamespace(
                                goals=["goal"],
                                facts=[],
                                decisions=[],
                                plans=[],
                            )
                        )
                    )
                ]
            )

    class FakeMistral:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("src.ai_devs_core.session.Mistral", FakeMistral)

    agent = SimpleNamespace(
        model_id="mistral-large-latest",
        config=SimpleNamespace(MISTRAL_API_KEY="test-key"),
    )
    session = BaseSessionManager(agent=agent, system_prompt="system")

    summary = session._summarize([{"role": "user", "content": "remember this"}])

    assert summary == "Goals: goal"
    assert calls["client_kwargs"]["timeout_ms"] == MISTRAL_TIMEOUT_MS
    assert calls["parse_kwargs"]["timeout_ms"] == MISTRAL_TIMEOUT_MS
    assert calls["parse_kwargs"]["max_tokens"] == MISTRAL_CHAT_MAX_TOKENS
