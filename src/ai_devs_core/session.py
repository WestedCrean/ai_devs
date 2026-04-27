import json
from loguru import logger
from pydantic import BaseModel
from typing import Protocol

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
from src.ai_devs_core.utils import count_tokens

_CONTEXT_SIZES: dict[str, int] = {
    "mistral-small-latest": 131_072,
    "mistral-small-3-1": 131_072,
    "mistral-medium-latest": 131_072,
    "mistral-large-latest": 131_072,
    "mistral-small-4-119b": 262_144,
    "mistral-small-4-119b-2603": 262_144,
}


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


class SessionSummary(BaseModel):
    goals: list[str] = []
    facts: list[str] = []
    decisions: list[str] = []
    plans: list[str] = []

class SessionManager(Protocol):
    def get_messages(self) -> list[dict]: ...
    def add_user_message(self, msg:str) -> None: ...
    def add_agent_message(self, msg: str) -> None: ...
    def add_tool_call_message(self, full_content: str, tool_calls_list: list) -> None: ...
    def add_tool_result_message(self, tool_call_id: str, content: str) -> None: ...
    def compress(self) -> None: ...

class BaseSessionManager:
    def __init__(self, agent, system_prompt: str):
        self._agent: BaseAgent = agent
        # Store as plain dicts - compatible with the Mistral SDK's chat methods.
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

    def get_messages(self) -> list[dict]:
        if self.occupancy > self.context_size * 0.4:
            logger.info(
                "Context occupancy at %d tokens, compressing session history",
                self.occupancy,
            )
            self.compress()

        return self.messages

    def add_user_message(self, msg: str):
        self.messages.append({"role": "user", "content": msg})

    def add_agent_message(self, msg: str):
        self.messages.append({"role": "assistant", "content": msg})

    def add_tool_call_message(self, full_content: str, tool_calls_list: list):
        """Append the assistant message that requested tool calls."""
        self.messages.append(
            {
                "role": "assistant",
                "content": full_content or None,
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

    def add_tool_result_message(self, tool_call_id: str, content: str):
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

        client = Mistral(api_key=self._agent.config.MISTRAL_API_KEY)
        response = client.chat.parse(
            model=self._agent.model_id,
            messages=[{"role": "system", "content": prompt}],
            response_format=SessionSummary,
        )
        parsed: SessionSummary = response.choices[0].message.parsed
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

    def compress(self):
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
    def compress(self):
        """Reduce context by keeping the system prompt and the 10 most recent messages.

        Keeps an even tail count to avoid splitting assistant/user turn pairs.
        """
        if sum([count_tokens(m.content) for m in self.messages]) <= 1_500:  # ty:ignore[unresolved-attribute]
            return

        system = self.messages[0]
        recent = self.messages[-10:]
        old = self.messages[1:-10]

        summary = super()._summarize(old)

        self.messages = [
            system,
            {"role": "assistant", "content": f"Conversation memory:\n{summary}"},
            *recent,
        ]
