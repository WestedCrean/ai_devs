import json

from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

_CONTEXT_SIZES: dict[str, int] = {
    "mistral-small-latest": 32_768,
    "mistral-small-3-1": 32_768,
    "mistral-medium-latest": 131_072,
    "mistral-large-latest": 131_072,
    "mistral-small-4-119b": 131_072,
    "mistral-small-4-119b-2603": 131_072,
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
            return AssistantMessage(content=msg.get("content"), tool_calls=tool_calls)
        return AssistantMessage(content=msg.get("content"))
    if role == "tool":
        return ToolMessage(content=msg["content"], tool_call_id=msg.get("tool_call_id"))
    raise ValueError(f"Unknown role: {role}")


class SessionManager:
    def __init__(self, agent, system_prompt: str):
        self._agent = agent
        # Store as plain dicts - compatible with the Mistral SDK's chat methods.
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]

    def get_messages(self) -> list[dict]:
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

    def get_context_size(self) -> int:
        """Return the model's maximum context window in tokens."""
        return _CONTEXT_SIZES.get(self._agent.model_id, 32_768)

    def calculate_occupancy(self) -> int:
        """Return the number of tokens currently used by this session."""
        tokenizer = MistralTokenizer.v3()
        mistral_messages = [_to_mistral_common(m) for m in self.messages]
        # continue_final_message=True avoids the validator rejecting assistant-last sessions.
        request = ChatCompletionRequest(
            messages=mistral_messages, continue_final_message=True
        )
        tokenized = tokenizer.encode_chat_completion(request)
        return len(tokenized.tokens)

    def compress(self):
        """Reduce context by keeping the system prompt and the 10 most recent messages.

        Keeps an even tail count to avoid splitting assistant/user turn pairs.
        """
        if len(self.messages) <= 11:
            return
        self.messages = [self.messages[0]] + self.messages[-10:]
