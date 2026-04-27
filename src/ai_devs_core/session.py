import json
from loguru import logger
from pydantic import BaseModel
from typing import Any, Protocol, cast

from mistralai.client import Mistral

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


from src.ai_devs_core.agent import BaseAgent
from src.ai_devs_core.agent import MISTRAL_CHAT_MAX_TOKENS
from src.ai_devs_core.agent import MISTRAL_TIMEOUT_MS
from src.ai_devs_core.memory import ObservedMemory
from src.ai_devs_core.utils import count_tokens

_CONTEXT_SIZES: dict[str, int] = {
    "mistral-small-latest": 131_072,
    "mistral-small-3-1": 131_072,
    "mistral-medium-latest": 131_072,
    "mistral-large-latest": 131_072,
    "mistral-small-4-119b": 262_144,
    "mistral-small-4-119b-2603": 262_144,
}

_MEMORY_MESSAGE_PREFIX = "<observational_memory>"


def _content_to_text(content: object) -> str:
    """Convert provider content chunks into plain text for stored history."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part for part in (_content_to_text(item) for item in content) if part
        )
    if isinstance(content, dict):
        data = cast(dict[str, Any], content)
        if isinstance(data.get("text"), str):
            return data["text"]
        if isinstance(data.get("content"), str):
            return data["content"]
        if isinstance(data.get("thinking"), list):
            return _content_to_text(data["thinking"])
    return str(content)


def _to_mistral_common(msg: dict):
    """Convert a message dict to a mistral_common message object for tokenization."""
    role = msg["role"]
    if role == "system":
        return SystemMessage(content=msg["content"])
    if role == "user":
        return UserMessage(content=msg["content"])
    if role == "assistant":
        if msg.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    function=FunctionCall(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
                for tc in msg["tool_calls"]
            ]
            return AssistantMessage(content=None, tool_calls=tool_calls)
        return AssistantMessage(content=msg.get("content"))
    if role == "tool":
        return ToolMessage(content=msg["content"], tool_call_id=msg.get("tool_call_id"))
    raise ValueError(f"Unknown role: {role}")


def _normalize_message(msg: dict) -> dict:
    """Return a Mistral-valid dict-backed chat message."""
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        return {**msg, "content": None}
    if "content" in msg:
        content = _content_to_text(msg.get("content"))
        if msg.get("role") == "assistant" and not content:
            content = "[non-text assistant content omitted]"
        return {**msg, "content": content}
    return msg


class SessionSummary(BaseModel):
    goals: list[str] = []
    facts: list[str] = []
    decisions: list[str] = []
    plans: list[str] = []


class SessionManager(Protocol):
    def get_messages(self) -> list[dict]: ...
    def add_user_message(self, msg: str) -> None: ...
    def add_agent_message(self, msg: str) -> None: ...
    def add_tool_call_message(self, full_content: str, tool_calls_list: list) -> None: ...
    def add_tool_result_message(self, tool_call_id: str, content: str) -> None: ...
    def compress(self) -> None: ...

class BaseSessionManager:
    def __init__(
        self,
        agent,
        system_prompt: str,
        max_context_tokens: int = 20_000,
    ):
        self._agent: BaseAgent = agent
        self.max_context_tokens = max_context_tokens
        # Store as plain dicts - compatible with the Mistral SDK's chat methods.
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

    def get_messages(self) -> list[dict]:
        self._normalize_messages()
        if self.occupancy > self.max_context_tokens:
            logger.info(
                "Context occupancy at %d tokens exceeds %d, compressing session history",
                self.occupancy,
                self.max_context_tokens,
            )
            self.compress()

        return self.messages

    def _normalize_messages(self) -> None:
        """Normalize stored chat history before tokenization or provider calls."""
        self.messages = [_normalize_message(message) for message in self.messages]

    def add_user_message(self, msg: str) -> None:
        self.messages.append({"role": "user", "content": _content_to_text(msg)})

    def add_agent_message(self, msg: object) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": _content_to_text(msg)
                or "[non-text assistant content omitted]",
            }
        )

    def add_tool_call_message(self, full_content: str, tool_calls_list: list) -> None:
        """Append the assistant message that requested tool calls."""
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": (
                                tc.function.arguments
                                if isinstance(tc.function.arguments, str)
                                else json.dumps(tc.function.arguments)
                            ),
                        },
                    }
                    for tc in tool_calls_list
                ],
            }
        )

    def add_tool_result_message(self, tool_call_id: str, content: str) -> None:
        """Append a tool result message."""
        self.messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    @property
    def context_size(self) -> int:
        """Return the model's maximum context window in tokens."""
        return _CONTEXT_SIZES.get(self._agent.model_id, 32_768)

    @property
    def occupancy(self) -> int:
        """Return the number of tokens currently used by this session."""
        self._normalize_messages()
        tokenizer = MistralTokenizer.v3()
        mistral_messages = [_to_mistral_common(m) for m in self.messages]
        # continue_final_message=True avoids the validator rejecting assistant-last sessions.
        request = ChatCompletionRequest(
            messages=mistral_messages, continue_final_message=False
        )
        tokenized = tokenizer.encode_chat_completion(request)
        return len(tokenized.tokens)

    def _summarize(self, messages: list[dict]) -> str:
        text = "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)

        prompt = f"""
            Summarize conversation into concise memory.

            Preserve:
            - goals
            - facts
            - decisions
            - plans

            Conversation:
            {text}
            """

        client = Mistral(
            api_key=self._agent.config.MISTRAL_API_KEY,
            timeout_ms=MISTRAL_TIMEOUT_MS,
        )
        response = client.chat.parse(
            model=self._agent.model_id,
            messages=[{"role": "system", "content": prompt}],
            response_format=SessionSummary,
            timeout_ms=MISTRAL_TIMEOUT_MS,
            max_tokens=MISTRAL_CHAT_MAX_TOKENS,
        )
        response_any = cast(Any, response)
        parsed = response_any.choices[0].message.parsed
        if parsed is None:
            return ""
        parts = []
        if parsed.goals:
            parts.append("Goals: " + "; ".join(parsed.goals))
        if parsed.facts:
            parts.append("Facts: " + "; ".join(parsed.facts))
        if parsed.decisions:
            parts.append("Decisions: " + "; ".join(parsed.decisions))
        if parsed.plans:
            parts.append("Plans: " + "; ".join(parsed.plans))
        return "\n".join(parts)

    def compress(self) -> None:
        """Reduce context by keeping the system prompt and the 10 most recent messages.

        Keeps an even tail count to avoid splitting assistant/user turn pairs.
        """
        if len(self.messages) < 12:
            return

        system = self.messages[0]
        recent = self.messages[-10:]
        old = self.messages[1:-10]

        summary = self._summarize(old)

        self.messages = [
            system,
            {"role": "assistant", "content": f"Conversation memory:\n{summary}"},
            *recent,
        ]

class RefinedSessionManager(BaseSessionManager):
    def __init__(
        self,
        agent,
        system_prompt: str,
        max_context_tokens: int = 1_500,
        recent_message_count: int = 10,
        memory: ObservedMemory | None = None,
    ):
        super().__init__(
            agent=agent,
            system_prompt=system_prompt,
            max_context_tokens=max_context_tokens,
        )
        self.recent_message_count = recent_message_count
        self.memory = memory or ObservedMemory(current_task=system_prompt.strip())

    def compress(self) -> None:
        """Compress older session context into observational memory."""
        if self._message_tokens(self.messages) <= self.max_context_tokens:
            return

        system = self.messages[0]
        session_messages = self._without_memory_messages(self.messages[1:])
        if len(session_messages) < 3:
            return

        recent_count = min(self.recent_message_count, len(session_messages))
        recent = session_messages[-recent_count:]
        old = session_messages[:-recent_count]

        if not old and len(session_messages) > 2:
            old = session_messages[:-2]
            recent = session_messages[-2:]

        if not old:
            return

        self.memory.observe_messages(self._agent, old)
        self.memory.raw_messages = []
        self.memory.reflect(self._agent)

        self.messages = [
            system,
            self._memory_message(),
            *recent,
        ]

    def _memory_message(self) -> dict:
        """Return the rendered memory block as a chat message."""
        return {
            "role": "assistant",
            "content": self.memory.get_memory_state(),
        }

    @staticmethod
    def _message_tokens(messages: list[dict]) -> int:
        """Return approximate token count for dict-backed chat messages."""
        return sum(count_tokens(str(message.get("content") or "")) for message in messages)

    @staticmethod
    def _without_memory_messages(messages: list[dict]) -> list[dict]:
        """Remove previously rendered observational memory messages."""
        return [
            message
            for message in messages
            if not str(message.get("content") or "").startswith(_MEMORY_MESSAGE_PREFIX)
        ]
