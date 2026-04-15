from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


from src.ai_devs_core import FAgent

_CONTEXT_SIZES: dict[str, int] = {
    "mistral-small-latest": 32_768,
    "mistral-small-3-1": 32_768,
    "mistral-medium-latest": 131_072,
    "mistral-large-latest": 131_072,
    "mistral-small-4-119b": 131_072,
    "mistral-small-4-119b-2603": 131_072,
}


class SessionManager:
    def __init__(self, agent: "FAgent", system_prompt: str):
        self._agent: FAgent = agent
        self.messages = [
            SystemMessage(content=system_prompt),
        ]

    def get_messages(self) -> list:
        return self.messages

    def add_user_message(self, msg: str):
        self.messages.append(UserMessage(content=msg))

    def add_agent_message(self, msg: str):
        self.messages.append(AssistantMessage(content=msg))

    def add_tool_call_message(self, full_content: str, tool_calls_list: list):
        """Append an assistant message that contains tool call requests."""
        tool_calls = [
            ToolCall(
                id=tc.id,
                function=FunctionCall(
                    name=tc.function.name,
                    arguments=(
                        tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else str(tc.function.arguments)
                    ),
                ),
            )
            for tc in tool_calls_list
        ]
        self.messages.append(
            AssistantMessage(content=full_content or None, tool_calls=tool_calls)
        )

    def add_tool_result_message(self, tool_call_id: str, content: str):
        """Append a tool result message."""
        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))

    def get_context_size(self) -> int:
        """Return the model's maximum context window in tokens."""
        return _CONTEXT_SIZES.get(self._agent.model_id, 32_768)

    def calculate_occupancy(self) -> int:
        """Return the number of tokens currently used by this session."""
        tokenizer = MistralTokenizer.v3()
        request = ChatCompletionRequest(messages=self.messages)
        tokenized = tokenizer.encode_chat_completion(request)
        return len(tokenized.tokens)

    def compress(self):
        """Reduce context by keeping the system prompt and the most recent 10 messages.

        Keeps an even number of tail messages to avoid splitting assistant/user turn pairs.
        """
        if len(self.messages) <= 11:
            return
        self.messages = [self.messages[0]] + self.messages[-10:]

    def get_session(self):
        return ChatCompletionRequest(messages=self.messages)
