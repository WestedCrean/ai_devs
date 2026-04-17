import os
import json
import time
import inspect
import re
import functools
from abc import ABC, abstractmethod
from typing import Callable, Any, Type
import pydantic
import polars as pl
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from mistralai.client.models import ToolMessage
from openai import OpenAI
from openrouter import OpenRouter
from loguru import logger
from langfuse import get_client as get_langfuse_client

from src.ai_devs_core.config import get_config
from src.ai_devs_core.job_client import RateLimiter, ErrorClassifier


def _retry_delay(e: Exception, attempt: int, base_max: float = 30) -> float:
    """Return retry delay in seconds. 429 rate-limit errors always wait at least 5s."""
    msg = str(e).lower()
    is_rate_limit = "429" in msg or "rate limit" in msg or "rate_limited" in msg
    if is_rate_limit:
        return max(5.0, min(2**attempt + (0.1 * (attempt + 1)), base_max))
    return min(2**attempt + (0.1 * (attempt + 1)), base_max)


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    if not docstring:
        return {}
    param_desc = {}
    pattern = re.compile(r"^[ \t]*(\w+)[ \t]*(?:\([^)]+\))?:[ \t]+(.*)$", re.MULTILINE)
    for match in pattern.finditer(docstring):
        param_desc[match.group(1)] = match.group(2).strip()
    return param_desc


def tool_logging(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        params = ", ".join(
            [repr(a) for a in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        )
        logger.info(f"tool call: {func.__name__}({params}) -> {result}")
        return result

    return wrapper


def verify_model_exists(model_id: str) -> bool:
    config = get_config()
    client = Mistral(api_key=config.MISTRAL_API_KEY)
    try:
        client.models.retrieve(model_id=model_id)
        return True
    except SDKError:
        return False


class BaseAgent(ABC):
    """Abstract base class for all LLM agent implementations."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.config = get_config()
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = self.config.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = self.config.LANGFUSE_BASE_URL
        self.langfuse = get_langfuse_client()

    @abstractmethod
    def chat_completion(
        self,
        chat_history: list[dict] = [],
        session_manager=None,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps: int = 4,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        stream: bool = False,
    ): ...

    def _generate_tool_definition(self, func: Callable) -> dict[str, Any]:
        """Convert a callable to an OpenAI-compatible function tool definition."""
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""
        main_desc = (
            docstring.split("\n\n")[0].replace("\n", " ").strip()
            if docstring
            else f"Executes {func.__name__}"
        )

        type_mapping = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        def _get_json_type(annotation) -> str:
            origin = getattr(annotation, "__origin__", None)
            if origin is list:
                return "array"
            if origin is dict:
                return "object"
            return type_mapping.get(annotation, "string")

        param_descriptions = _parse_docstring_params(docstring)
        properties, required_params = {}, []

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "*args", "**kwargs"):
                continue
            properties[name] = {
                "type": _get_json_type(param.annotation),
                "description": param_descriptions.get(name, f"The {name} parameter."),
            }
            if param.default is inspect.Parameter.empty:
                required_params.append(name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": main_desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                },
            },
        }

    def run_infer_on_each_row(
        self,
        df: pl.DataFrame,
        columns: list[str],
        output_col: str,
        query: str,
        tools: list[Callable],
        response_schema: Type[pydantic.BaseModel] = None,
    ) -> pl.DataFrame:
        """
        Runs model inference sequentially one by one on each row

        df: polars.DataFrame - dataframe to transform
        columns: list[str] - fields to pass to llm model
        query: str - query to pass to llm model
        tools: list[Callable] - functions llm model can use
        """
        results = []
        for row in df.iter_rows(named=True):
            output = self.chat_completion(
                chat_history=[
                    {
                        "role": "user",
                        "content": query.format(**{c: row[c] for c in columns}),
                    }
                ],
                tools=tools,
                response_schema=response_schema,
            )
            results.append(output.choices[0].message.content)
        return df.with_columns(pl.Series(name=output_col, values=results))

    def batch_job(self, *args, **kwargs):
        raise NotImplementedError(
            f"Batch jobs are not implemented for {type(self).__name__}"
        )


class ORAgent(BaseAgent):
    def __init__(self, model_id: str = "openai/gpt-4o"):
        super().__init__(model_id)

    def chat_completion(
        self,
        chat_history: list[dict] = [],
        session_manager=None,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps: int = 15,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        stream: bool = False,
        max_retries: int = 5,
    ):
        messages = list(chat_history)
        openrouter_tools = (
            [self._generate_tool_definition(t) for t in tools] if tools else None
        )
        tool_map = {t.__name__: t for t in tools} if tools else {}

        with self.langfuse.start_as_current_observation(
            as_type="generation",
            name=f"openrouter/{self.model_id}",
        ) as obs:
            obs.update(
                input=messages,
                model=self.model_id,
                metadata=(
                    {"tools": [t["function"]["name"] for t in openrouter_tools]}
                    if openrouter_tools
                    else {}
                ),
            )

            response = None
            with OpenRouter(api_key=self.config.OPENROUTER_API_KEY) as client:
                for step in range(max_steps):
                    msg = None
                    for attempt in range(max_retries + 1):
                        try:
                            kwargs = dict(model=self.model_id, messages=messages)
                            if openrouter_tools:
                                kwargs["tools"] = openrouter_tools
                                kwargs["tool_choice"] = "auto"

                            response = client.chat.send(**kwargs)
                            msg = response.choices[0].message

                            msg_dict: dict = {
                                "role": "assistant",
                                "content": msg.content or None,
                            }
                            if msg.tool_calls:
                                msg_dict["tool_calls"] = [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in msg.tool_calls
                                ]
                                if session_manager:
                                    session_manager.add_tool_call_message(
                                        msg.content or "", msg.tool_calls
                                    )
                            messages.append(msg_dict)
                            break

                        except Exception as e:
                            if (
                                ErrorClassifier.is_retryable(e)
                                and attempt < max_retries
                            ):
                                delay = _retry_delay(e, attempt, base_max=10)
                                logger.warning(
                                    f"Retryable error step {step}, attempt {attempt + 1}/{max_retries + 1}: {e}"
                                )
                                time.sleep(delay)
                                continue
                            raise

                    if msg is None or not msg.tool_calls:
                        break

                    for tc in msg.tool_calls:
                        func = tool_map[tc.function.name]
                        args = json.loads(tc.function.arguments)
                        if on_tool_call:
                            on_tool_call(tc.function.name, args)
                        result = func(**args)
                        if on_tool_result:
                            on_tool_result(tc.function.name, str(result))
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(result),
                            }
                        )
                        if session_manager:
                            session_manager.add_tool_result_message(tc.id, str(result))

                if response_schema:
                    from types import SimpleNamespace

                    last = messages[-1]
                    last_role = (
                        last.get("role")
                        if isinstance(last, dict)
                        else getattr(last, "role", None)
                    )
                    parse_messages = (
                        messages[:-1] if last_role == "assistant" else messages
                    )
                    parse_response = client.chat.send(
                        model=self.model_id,
                        messages=parse_messages,
                        max_tokens=8000,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": response_schema.__name__,
                                "schema": response_schema.model_json_schema(),
                            },
                        },
                    )
                    raw_content = parse_response.choices[0].message.content or ""
                    parsed = response_schema.model_validate_json(raw_content)
                    response = SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content=raw_content, parsed=parsed
                                )
                            )
                        ],
                        usage=parse_response.usage,
                    )

            content = response.choices[0].message.content or ""
            if on_token and content:
                on_token(content)

            obs.update(
                output=content,
                usage_details={
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                },
            )

        return response


class OAgent(BaseAgent):
    def __init__(
        self,
        model_id: str = "gpt-3.5-turbo",
        api_base: str = None,
        api_key: str = None,
    ):
        super().__init__(model_id)
        self.api_base = api_base
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate=4, max_tokens=2000)

    def chat_completion(
        self,
        chat_history: list[dict] = [],
        session_manager=None,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps: int = 4,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        stream: bool = False,
        tool_max_retries: int = 5,
    ):
        messages = list(chat_history)
        openai_tools = (
            [self._generate_tool_definition(t) for t in tools] if tools else None
        )
        tool_map = {t.__name__: t for t in tools} if tools else {}

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for _ in range(max_steps):
            kwargs = dict(model=self.model_id, messages=messages)
            if openai_tools:
                kwargs["tools"] = openai_tools
                kwargs["tool_choice"] = "auto"
            response = client.chat.completions.create(**kwargs)
            logger.info(response)
            msg = response.choices[0].message
            messages.append(msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                func = tool_map[tc.function.name]
                args = json.loads(tc.function.arguments)
                if on_tool_call:
                    on_tool_call(tc.function.name, args)
                result = func(**args)
                if on_tool_result:
                    on_tool_result(tc.function.name, str(result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                )

        if response_schema:
            last = messages[-1]
            last_role = (
                last.get("role")
                if isinstance(last, dict)
                else getattr(last, "role", None)
            )
            parse_messages = messages[:-1] if last_role == "assistant" else messages
            schema_response = client.chat.completions.create(
                model=self.model_id,
                messages=parse_messages,
                response_format={"type": "json_object"},
            )
            return schema_response

        return response


class FAgent(BaseAgent):
    def __init__(self, model_id: str):
        if not verify_model_exists(model_id=model_id):
            raise Exception(f"Model {model_id} not available in Mistral API")
        super().__init__(model_id)

    def _stream_step(
        self,
        client: Any,
        kwargs: dict,
        on_token: Callable[[str], None] | None,
    ) -> tuple[str, list | None]:
        """Stream one LLM step. Returns (full_content, tool_calls_list)."""
        full_content = ""
        tool_calls_list = None

        with client.chat.stream(**kwargs) as s:
            for event in s:
                delta = event.data.choices[0].delta
                if isinstance(delta.content, str) and delta.content:
                    full_content += delta.content
                    if on_token:
                        on_token(delta.content)
                if isinstance(delta.tool_calls, list) and delta.tool_calls:
                    tool_calls_list = delta.tool_calls

        return full_content, tool_calls_list

    def chat_completion(
        self,
        chat_history: list[dict] = [],
        session_manager=None,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps: int = 4,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        stream: bool = False,
    ):
        from types import SimpleNamespace

        messages = list(chat_history)
        mistral_tools = (
            [self._generate_tool_definition(t) for t in tools] if tools else None
        )
        tool_map = {t.__name__: t for t in tools} if tools else {}

        client = Mistral(api_key=self.config.MISTRAL_API_KEY)

        max_retries = 5
        for step in range(max_steps):
            time.sleep(1.6)
            kwargs = dict(
                model=self.model_id,
                messages=messages,
            )
            if self.model_id in ("mistral-small-latest",):
                kwargs["reasoning_effort"] = "high"
            if mistral_tools:
                kwargs["tools"] = mistral_tools
                kwargs["tool_choice"] = "auto"
                kwargs["parallel_tool_calls"] = False

            for attempt in range(max_retries + 1):
                try:
                    if stream:
                        full_content, tool_calls_list = self._stream_step(
                            client, kwargs, on_token
                        )
                        asst_msg: dict = {
                            "role": "assistant",
                            "content": full_content or None,
                        }
                        if tool_calls_list:
                            asst_msg["tool_calls"] = [
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
                            ]
                            if session_manager:
                                session_manager.add_tool_call_message(
                                    full_content, tool_calls_list
                                )
                        messages.append(asst_msg)
                        response = SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(content=full_content)
                                )
                            ],
                            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                        )
                        msg_tool_calls = tool_calls_list
                    else:
                        response = client.chat.complete(**kwargs)
                        msg = response.choices[0].message
                        messages.append(msg)
                        msg_tool_calls = msg.tool_calls
                        if msg_tool_calls and session_manager:
                            session_manager.add_tool_call_message(
                                msg.content or "", msg_tool_calls
                            )
                    break  # success
                except Exception as e:
                    if ErrorClassifier.is_retryable(e) and attempt < max_retries:
                        delay = _retry_delay(e, attempt, base_max=30)
                        logger.warning(
                            f"Retryable error step {step}, attempt {attempt + 1}/{max_retries + 1}: {e}"
                        )
                        time.sleep(delay)
                        continue
                    raise

            if not msg_tool_calls:
                break

            for tc in msg_tool_calls:
                func = tool_map[tc.function.name]
                args_raw = tc.function.arguments
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                if on_tool_call:
                    on_tool_call(tc.function.name, args)
                result = func(**args)
                if on_tool_result:
                    on_tool_result(tc.function.name, str(result))
                tool_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tc.id,
                )
                messages.append(tool_msg)
                if session_manager:
                    session_manager.add_tool_result_message(tc.id, str(result))

        if response_schema:
            last = messages[-1]
            last_role = (
                last.get("role")
                if isinstance(last, dict)
                else getattr(last, "role", None)
            )
            parse_messages = messages[:-1] if last_role == "assistant" else messages
            response = client.chat.parse(
                model=self.model_id,
                messages=parse_messages,
                response_format=response_schema,
            )

        return response

    def _generate_reflection(
        self,
        original_message: str,
        current_response: str,
        reflection_model: str | None = None,
        reflection_prompt: str = "Verify the following response for correctness and point out obvious errors",
        reflection_depth: int = 1,
    ) -> str:
        model_id = reflection_model or self.model_id
        reflection_agent = FAgent(model_id=model_id) if reflection_model else self

        reflection_input = f"""{reflection_prompt}

        Original question: {original_message}

        Current response: {current_response}

        Please verify for obvious errors"""

        reflection_response = reflection_agent.chat_completion(
            chat_history=[{"role": "user", "content": reflection_input}],
            max_steps=reflection_depth,
        )
        return reflection_response.choices[0].message.content

    def chat_completion_with_reflect(
        self,
        message: str,
        chat_history: list[dict] = [],
        streaming_response: bool = False,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps: int = 4,
        max_reflections: int = 1,
        reflection_model: str | None = None,
        reflection_prompt: str = "Analyze the following response for accuracy, completeness, and quality. Provide constructive criticism and suggestions for improvement:",
    ):
        """Enhanced chat completion with reflection/critique loop.

        Args:
            message: User message/content
            chat_history: Prior conversation context
            streaming_response: Not supported in reflection mode (ignored)
            tools: List of callable tools the agent can use
            response_schema: Pydantic model for structured output
            max_steps: Maximum tool calling iterations per response generation
            max_reflections: Maximum number of reflection/critique iterations
            reflection_model: Optional different model for reflection/critique
            reflection_prompt: Custom prompt for the reflection agent

        Returns:
            Chat completion response with refined output
        """
        if streaming_response:
            logger.warning(
                "Streaming response not supported in reflection mode, disabling"
            )

        original_message = message
        current_response_content = ""
        final_response = None
        critique = ""

        for reflection_iteration in range(max_reflections):
            logger.info(
                f"Reflection iteration {reflection_iteration + 1}/{max_reflections}"
            )

            if reflection_iteration == 0:
                response = self.chat_completion(
                    chat_history=list(chat_history)
                    + [{"role": "user", "content": message}],
                    tools=tools,
                    response_schema=None,
                    max_steps=2,
                )
            else:
                critique_context = f"""Original question: {original_message}

Previous response: {current_response_content}

Critique and suggestions: {critique}

Please generate an improved response incorporating the feedback:"""

                response = self.chat_completion(
                    chat_history=list(chat_history)
                    + [{"role": "user", "content": critique_context}],
                    tools=tools,
                    response_schema=None,
                    max_steps=max_steps,
                )

            current_response_content = response.choices[0].message.content

            if reflection_iteration < max_reflections - 1:
                critique = self._generate_reflection(
                    original_message=original_message,
                    current_response=current_response_content,
                    reflection_model=reflection_model,
                    reflection_prompt=reflection_prompt,
                    reflection_depth=2,
                )
                logger.info(f"Reflection critique: {critique[:100]}...")

                critique_lower = critique.lower()
                should_continue = any(
                    keyword in critique_lower
                    for keyword in [
                        "improve",
                        "missing",
                        "incorrect",
                        "incomplete",
                        "better",
                        "add",
                        "change",
                        "modify",
                        "enhance",
                        "revise",
                    ]
                )

                if not should_continue:
                    logger.info(
                        "Critique suggests response is satisfactory, ending reflection early"
                    )
                    break

            final_response = response

        if response_schema:
            schema_response = self.chat_completion(
                chat_history=list(chat_history)
                + [{"role": "user", "content": original_message}],
                tools=tools,
                response_schema=response_schema,
                max_steps=max_steps,
            )
            return schema_response

        return final_response
