from types import SimpleNamespace

from src.ai_devs_core.session import BaseSessionManager


class FakeAgent:
    """Test double for session tokenization."""

    model_id = "mistral-small-latest"


def test_tool_call_messages_store_tool_calls_without_content() -> None:
    """Tool-call messages should be valid for Mistral tokenization."""
    manager = BaseSessionManager(agent=FakeAgent(), system_prompt="System prompt.")
    tool_call = SimpleNamespace(
        id="call12345",
        function=SimpleNamespace(name="lookup", arguments='{"value": "abc"}'),
    )

    manager.add_tool_call_message("I will call a tool.", [tool_call])
    manager.add_tool_result_message("call12345", "found abc")

    assert manager.messages[-2]["content"] is None
    assert manager.occupancy > 0


def test_get_messages_normalizes_existing_tool_call_content() -> None:
    """Existing assistant messages with content and tool calls should be repaired."""
    manager = BaseSessionManager(agent=FakeAgent(), system_prompt="System prompt.")
    manager.messages.extend(
        [
            {
                "role": "assistant",
                "content": "I will call a tool.",
                "tool_calls": [
                    {
                        "id": "call12345",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"value": "abc"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call12345", "content": "found abc"},
        ]
    )

    messages = manager.get_messages()

    assert messages[-2]["content"] is None
    assert manager.occupancy > 0


def test_get_messages_normalizes_structured_assistant_content() -> None:
    """Structured provider content should be converted to plain text."""
    manager = BaseSessionManager(agent=FakeAgent(), system_prompt="System prompt.")
    manager.messages.append(
        {
            "role": "assistant",
            "content": [
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
            ],
        }
    )
    manager.messages.append({"role": "user", "content": "continue"})

    messages = manager.get_messages()

    assert messages[-2]["content"] == "I need to inspect more log lines."
    assert manager.occupancy > 0
