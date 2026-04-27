from types import SimpleNamespace

import httpx
from pydantic import BaseModel

from src.ai_devs_core import agent as agent_module
from src.ai_devs_core.agent import MISTRAL_CHAT_MAX_TOKENS, OAgent, ORAgent


class ContractSchema(BaseModel):
    """Structured response used by agent contract tests."""

    answer: str


class FakeSessionManager:
    """Capture session-manager interactions."""

    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, list]] = []
        self.tool_results: list[tuple[str, str]] = []

    def add_tool_call_message(self, full_content: str, tool_calls_list: list) -> None:
        """Store assistant tool-call messages."""
        self.tool_calls.append((full_content, tool_calls_list))

    def add_tool_result_message(self, tool_call_id: str, content: str) -> None:
        """Store tool result messages."""
        self.tool_results.append((tool_call_id, content))


class FakeObservation:
    """No-op Langfuse observation context manager."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs) -> None:
        """Accept observation updates."""


class FakeLangfuse:
    """No-op Langfuse client."""

    def start_as_current_observation(self, **kwargs):
        """Return a no-op observation context manager."""
        return FakeObservation()


def _message(content: str, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _response(content: str, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=_message(content, tool_calls))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )


def _tool_call(arguments: str = '{"value": "abc"}') -> SimpleNamespace:
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="lookup", arguments=arguments),
    )


def lookup(value: str) -> str:
    """Lookup test values."""
    return f"found:{value}"


def _new_oagent() -> OAgent:
    agent = OAgent.__new__(OAgent)
    agent.model_id = "gpt-test"
    agent.config = SimpleNamespace(OPENAI_API_KEY="openai-key")
    agent.api_key = None
    agent.api_base = None
    return agent


def _new_oragent() -> ORAgent:
    agent = ORAgent.__new__(ORAgent)
    agent.model_id = "openai/gpt-test"
    agent.config = SimpleNamespace(OPENROUTER_API_KEY="openrouter-key")
    agent.langfuse = FakeLangfuse()
    return agent


def test_oagent_matches_fagent_tool_session_callback_contract(monkeypatch) -> None:
    """OAgent should support FAgent-style tool loop callbacks and session updates."""
    calls: list[dict] = []
    responses = [_response("", [_tool_call()]), _response("done")]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)

    tokens: list[str] = []
    tool_calls: list[tuple[str, dict]] = []
    tool_results: list[tuple[str, str]] = []
    session = FakeSessionManager()

    response = _new_oagent().chat_completion(
        chat_history=[{"role": "user", "content": "use a tool"}],
        session_manager=session,
        tools=[lookup],
        on_tool_call=lambda name, args: tool_calls.append((name, args)),
        on_tool_result=lambda name, result: tool_results.append((name, result)),
        on_token=tokens.append,
        stream=True,
    )

    assert response.choices[0].message.content == "done"
    assert calls[0]["max_tokens"] == MISTRAL_CHAT_MAX_TOKENS
    assert calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "found:abc",
    }
    assert session.tool_calls[0][1][0].id == "call-1"
    assert session.tool_results == [("call-1", "found:abc")]
    assert tool_calls == [("lookup", {"value": "abc"})]
    assert tool_results == [("lookup", "found:abc")]
    assert tokens == ["done"]


def test_oragent_matches_fagent_tool_session_callback_contract(monkeypatch) -> None:
    """ORAgent should support FAgent-style tool loop callbacks and session updates."""
    calls: list[dict] = []
    responses = [_response("", [_tool_call()]), _response("done")]

    class FakeChat:
        def send(self, **kwargs):
            calls.append(kwargs)
            return responses.pop(0)

    class FakeOpenRouter:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(agent_module, "OpenRouter", FakeOpenRouter)

    tokens: list[str] = []
    session = FakeSessionManager()

    response = _new_oragent().chat_completion(
        chat_history=[{"role": "user", "content": "use a tool"}],
        session_manager=session,
        tools=[lookup],
        on_token=tokens.append,
        stream=True,
    )

    assert response.choices[0].message.content == "done"
    assert calls[0]["max_tokens"] == MISTRAL_CHAT_MAX_TOKENS
    assert calls[1]["messages"][-1]["content"] == "found:abc"
    assert session.tool_calls[0][1][0].id == "call-1"
    assert session.tool_results == [("call-1", "found:abc")]
    assert tokens == ["done"]


def test_oagent_retries_read_timeout_and_returns_parsed_schema(monkeypatch) -> None:
    """OAgent should retry transient timeouts and normalize parsed schema output."""
    calls: list[dict] = []
    responses = [
        httpx.ReadTimeout("The read operation timed out"),
        _response("ready"),
        _response('{"answer": "parsed"}'),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(agent_module, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

    response = _new_oagent().chat_completion(
        chat_history=[{"role": "user", "content": "json"}],
        response_schema=ContractSchema,
    )

    assert len(calls) == 3
    assert calls[-1]["response_format"]["type"] == "json_schema"
    assert response.choices[0].message.parsed == ContractSchema(answer="parsed")


def test_oragent_retries_read_timeout_and_returns_parsed_schema(monkeypatch) -> None:
    """ORAgent should retry transient timeouts and normalize parsed schema output."""
    calls: list[dict] = []
    responses = [
        httpx.ReadTimeout("The read operation timed out"),
        _response("ready"),
        _response('{"answer": "parsed"}'),
    ]

    class FakeChat:
        def send(self, **kwargs):
            calls.append(kwargs)
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class FakeOpenRouter:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(agent_module, "OpenRouter", FakeOpenRouter)
    monkeypatch.setattr(agent_module.time, "sleep", lambda _: None)

    response = _new_oragent().chat_completion(
        chat_history=[{"role": "user", "content": "json"}],
        response_schema=ContractSchema,
    )

    assert len(calls) == 3
    assert calls[-1]["response_format"]["type"] == "json_schema"
    assert response.choices[0].message.parsed == ContractSchema(answer="parsed")
