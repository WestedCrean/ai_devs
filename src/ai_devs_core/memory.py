from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.ai_devs_core.agent import BaseAgent
from src.ai_devs_core.utils import count_tokens

Message = dict[str, object]
Priority = Literal["HIGH", "MED", "LOW"]


class MemoryObservation(BaseModel):
    """A single text-first observation extracted from raw session messages."""

    observation_date: str
    referenced_date: str | None = None
    relative_date: str | None = None
    time: str | None = None
    priority: Priority = "MED"
    content: str

    model_config = ConfigDict(extra="forbid")

    def as_log_line(self) -> str:
        """Render the observation as a compact log-like memory line."""
        date_part = self.observation_date
        if self.time:
            date_part = f"{date_part} {self.time}"

        context_parts = []
        if self.referenced_date:
            context_parts.append(f"ref={self.referenced_date}")
        if self.relative_date:
            context_parts.append(f"rel={self.relative_date}")
        context = f" ({', '.join(context_parts)})" if context_parts else ""
        return f"- {date_part} [{self.priority}]{context} {self.content}"


class ObservationBatch(BaseModel):
    """Structured observer output."""

    observations: list[MemoryObservation] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ReflectionBatch(BaseModel):
    """Structured reflector output."""

    observations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ObservedMemory(BaseModel):
    """Observation-based memory for compressing long agent sessions."""

    observations: list[str] = Field(default_factory=list)
    raw_messages: list[Message] = Field(default_factory=list)
    current_task: str = ""
    observation_token_threshold: int = 40_000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add(self, observation: str) -> None:
        """Add one already-compressed observation."""
        if observation.strip():
            self.observations.append(observation.strip())

    def add_raw_message(self, message: Message) -> None:
        """Add one raw message awaiting observation."""
        self.raw_messages.append(message)

    def extend_raw_messages(self, messages: list[Message]) -> None:
        """Add raw messages awaiting observation."""
        self.raw_messages.extend(messages)

    @property
    def observation_tokens(self) -> int:
        """Return approximate token count of stored observations."""
        return count_tokens("\n".join(self.observations))

    @property
    def raw_tokens(self) -> int:
        """Return approximate token count of pending raw messages."""
        return count_tokens(self._messages_to_text(self.raw_messages))

    def get_memory_state(self) -> str:
        """Render memory as a stable context block for an LLM prompt."""
        observations = "\n".join(self.observations).strip() or "- No observations yet."
        task = self.current_task.strip() or "Not specified."
        raw = self._messages_to_text(self.raw_messages).strip() or "- No pending raw messages."
        return (
            "<observational_memory>\n"
            "<current_task>\n"
            f"{task}\n"
            "</current_task>\n"
            "<observations>\n"
            f"{observations}\n"
            "</observations>\n"
            "<unobserved_messages>\n"
            f"{raw}\n"
            "</unobserved_messages>\n"
            "</observational_memory>"
        )

    def observe_messages(
        self,
        agent: BaseAgent,
        messages: list[Message],
        observation_date: str | None = None,
    ) -> None:
        """Compress raw session messages into dated observations."""
        if not messages:
            return

        observed_at = observation_date or datetime.now().date().isoformat()
        prompt = f"""
You are an observer agent that converts raw conversation messages into durable memory.

Write compact observations that will help a future agent continue the task.
Preserve dates, times, IDs, filenames, decisions, user preferences, tool outcomes, errors,
verification feedback, and next actions. Do not preserve irrelevant wording.

Use:
- observation_date: {observed_at}
- referenced_date: concrete date mentioned by the message, or null
- relative_date: relative phrase such as "yesterday" or "in 1 week", or null
- time: HH:MM if available, or null
- priority: HIGH for task-critical facts, MED for useful facts, LOW for background

Current task:
{self.current_task or "Not specified."}

Raw messages:
{self._messages_to_text(messages)}
"""
        response = agent.chat_completion(
            chat_history=[{"role": "user", "content": prompt}],
            response_schema=ObservationBatch,
            max_steps=1,
        )
        parsed: ObservationBatch = response.choices[0].message.parsed
        for observation in parsed.observations:
            self.add(observation.as_log_line())

    def reflect(self, agent: BaseAgent, target_tokens: int | None = None) -> None:
        """Prune and merge observations when the observation block grows too large."""
        if not self.observations:
            return

        token_budget = target_tokens or self.observation_token_threshold
        if self.observation_tokens <= token_budget:
            return

        prompt = f"""
You are a memory reflector. Prune and merge observations while preserving the facts needed
to continue the task. Keep dates, times, IDs, filenames, decisions, tool results, errors,
verification feedback, and next actions. Remove duplicates and obsolete low-value details.

Target token budget: {token_budget}

Current task:
{self.current_task or "Not specified."}

Observations:
{chr(10).join(self.observations)}
"""
        response = agent.chat_completion(
            chat_history=[{"role": "user", "content": prompt}],
            response_schema=ReflectionBatch,
            max_steps=1,
        )
        parsed: ReflectionBatch = response.choices[0].message.parsed
        self.observations = [item.strip() for item in parsed.observations if item.strip()]

    @staticmethod
    def _messages_to_text(messages: list[Message]) -> str:
        """Render chat messages as compact text for observation prompts."""
        lines = []
        for message in messages:
            role = str(message.get("role", "unknown"))
            content = message.get("content")
            if content is None and message.get("tool_calls"):
                content = f"tool_calls={message['tool_calls']}"
            elif role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                content = f"tool_call_id={tool_call_id} result={content}"
            lines.append(f"{role}: {content or ''}")
        return "\n".join(lines)
