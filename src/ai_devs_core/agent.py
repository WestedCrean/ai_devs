import os
import json
import time
import inspect
import re
import functools
from typing import Callable, Any, Type
import pydantic
import polars as pl
from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from openai import OpenAI
from openrouter import OpenRouter
from loguru import logger
from langfuse import get_client as get_langfuse_client

from src.ai_devs_core.config import get_config
from src.ai_devs_core.job_client import RateLimiter, ErrorClassifier


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


class ORAgent:
    def __init__(self, model_id: str = "openai/gpt-5.2"):
        self.model_id = model_id
        self.config = get_config()
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = self.config.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = self.config.LANGFUSE_BASE_URL
        self.langfuse = get_langfuse_client()
        # Initialize rate limiter with reasonable defaults for OpenRouter
        # self.rate_limiter = RateLimiter(rate=4, max_tokens=2000)

    def chat_completion(
        self,
        message: str,
        streaming_response: bool = False,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps=15,
        max_retries=5,
    ):
        messages = [{"role": "user", "content": message}]
        openrouter_tools = (
            [self._generate_openrouter_tool(t) for t in tools] if tools else None
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

            with OpenRouter(api_key=self.config.OPENROUTER_API_KEY) as client:
                for step in range(max_steps):
                    # Apply rate limiting
                    # self.rate_limiter.wait()

                    for attempt in range(max_retries + 1):
                        try:
                            kwargs = dict(
                                model=self.model_id,
                                messages=messages,
                            )
                            if openrouter_tools:
                                kwargs["tools"] = openrouter_tools
                                kwargs["tool_choice"] = "auto"

                            response = client.chat.send(**kwargs)
                            msg = response.choices[0].message
                            messages.append(msg)

                            if not msg.tool_calls:
                                break

                            for tc in msg.tool_calls:
                                func = tool_map[tc.function.name]
                                args = json.loads(tc.function.arguments)
                                result = func(**args)
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tc.id,
                                        "content": str(result),
                                    }
                                )

                            # Successfully completed this step, break out of retry loop
                            break

                        except Exception as e:

                            # Check if this is a retryable error
                            if ErrorClassifier.is_retryable(e):
                                if attempt < max_retries:
                                    delay = min(
                                        2**attempt + (0.1 * (attempt + 1)), 10
                                    )  # Exponential backoff with jitter
                                    logger.warning(
                                        f"Retryable error in step {step}, attempt {attempt + 1}/{max_retries + 1}: {e}"
                                    )
                                    logger.info(f"Retrying in {delay:.2f} seconds...")
                                    time.sleep(delay)
                                    continue
                                else:
                                    logger.error(
                                        f"Max retries ({max_retries}) exceeded for retryable error: {e}"
                                    )
                                    raise
                            else:
                                # Non-retryable error, re-raise immediately
                                logger.error(f"Non-retryable error in step {step}: {e}")
                                raise

                # Final structured output call if schema provided
                if response_schema:
                    # chat.parse requires the last message to be user or tool,
                    # so drop the trailing assistant message from the loop.
                    last = messages[-1]
                    last_role = (
                        last.get("role")
                        if isinstance(last, dict)
                        else getattr(last, "role", None)
                    )
                    parse_messages = (
                        messages[:-1] if last_role == "assistant" else messages
                    )
                    # Apply rate limiting before final call
                    self.rate_limiter.wait()
                    response = client.chat.parse(
                        model=self.model_id,
                        messages=parse_messages,
                        response_format=response_schema,
                    )

                obs.update(
                    output=response.choices[0].message.content,
                    usage_details={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                    },
                )

        return response

    def _generate_openrouter_tool(self, func: Callable) -> dict[str, Any]:
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
        param_descriptions = _parse_docstring_params(docstring)

        properties, required_params = {}, []

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "*args", "**kwargs"):
                continue
            properties[name] = {
                "type": type_mapping.get(param.annotation, "string"),
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
        Runs model inference sequencially one by one on each row

        df: polars.DataFrame - dataframe to transform
        columns: list[str] - fields to pass to llm model
        query: str - query to pass to llm model
        tools: list[Callable] - functions llm model can use
        """

        results = []

        for row in df.iter_rows(named=True):
            output = self.chat_completion(
                query.format(**{c: row[c] for c in columns}),
                tools=tools,
                response_schema=response_schema,
            )
            results.append(output.choices[0].message.content)

        return df.with_columns(pl.Series(name=output_col, values=results))

    def batch_job(self, *args, **kwargs):
        raise NotImplementedError("Batch jobs are not implemented for ORAgent")


class OAgent:
    def __init__(
        self,
        model_id: str = "gpt-3.5-turbo",
        api_base: str = None,
        api_key: str = None,
    ):
        self.model_id = model_id
        self.config = get_config()

        # Set up environment variables for Langfuse
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = self.config.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = self.config.LANGFUSE_BASE_URL
        self.langfuse = get_langfuse_client()

        # Initialize OpenAI client with configurable endpoint
        self.api_base = api_base
        self.api_key = api_key

        # Initialize rate limiter with reasonable defaults
        self.rate_limiter = RateLimiter(rate=4, max_tokens=2000)

    def chat_completion(
        self,
        message: str,
        chat_history: list[dict] = [],
        streaming_response: bool = False,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps: int = 4,
        tool_max_retries: int = 5,
    ):
        messages = chat_history + [{"role": "user", "content": message}]
        openai_tools = [self._generate_openai_tool(t) for t in tools] if tools else None
        tool_map = {t.__name__: t for t in tools} if tools else {}

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        for step in range(max_steps):
            kwargs = dict(
                model=self.model_id,
                messages=messages,
            )
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
                result = func(**args)
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

    def _generate_openai_tool(self, func: Callable) -> dict[str, Any]:
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
        param_descriptions = _parse_docstring_params(docstring)

        properties, required_params = {}, []

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "*args", "**kwargs"):
                continue
            properties[name] = {
                "type": type_mapping.get(param.annotation, "string"),
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
                query.format(**{c: row[c] for c in columns}),
                tools=tools,
                response_schema=response_schema,
            )
            results.append(output.choices[0].message.content)

        return df.with_columns(pl.Series(name=output_col, values=results))

    def batch_job(self, *args, **kwargs):
        raise NotImplementedError("Batch jobs are not implemented for OAgent")


class FAgent:
    def __init__(self, model_id: str):
        if not verify_model_exists(model_id=model_id):
            raise Exception(f"Model {model_id} not available in Mistral API")
        self.model_id = model_id
        self.config = get_config()
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = self.config.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = self.config.LANGFUSE_BASE_URL
        self.langfuse = get_langfuse_client()

    def chat_completion(
        self,
        message: str,
        chat_history: list[dict] = [],
        streaming_response: bool = False,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps=4,
    ):
        messages = chat_history + [{"role": "user", "content": message}]
        mistral_tools = (
            [self._generate_mistral_tool(t) for t in tools] if tools else None
        )
        tool_map = {t.__name__: t for t in tools} if tools else {}

        with self.langfuse.start_as_current_observation(
            as_type="generation",
            name=f"mistral/{self.model_id}",
        ) as obs:
            obs.update(
                input=messages,
                model=self.model_id,
                metadata=(
                    {"tools": [t["function"]["name"] for t in mistral_tools]}
                    if mistral_tools
                    else {}
                ),
            )

            client = Mistral(api_key=self.config.MISTRAL_API_KEY)

            for _ in range(max_steps):
                time.sleep(0.5)
                kwargs = dict(
                    model=self.model_id,
                    messages=messages,
                )
                if mistral_tools:
                    kwargs["tools"] = mistral_tools
                    kwargs["tool_choice"] = "auto"

                response = client.chat.complete(**kwargs)
                msg = response.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    break

                for tc in msg.tool_calls:
                    func = tool_map[tc.function.name]
                    args = json.loads(tc.function.arguments)
                    result = func(**args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        }
                    )

            # Final structured output call if schema provided
            if response_schema:
                # chat.parse requires the last message to be user or tool,
                # so drop the trailing assistant message from the loop.
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

            obs.update(
                output=response.choices[0].message.content,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                },
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
        """Generate a critique/reflection on the current response.

        Args:
            original_message: The original user message
            current_response: The response to be critiqued
            reflection_model: Optional different model for reflection
            reflection_prompt: Custom prompt for the reflection agent

        Returns:
            str: Critique and suggestions for improvement
        """
        model_id = reflection_model or self.model_id

        # Create reflection agent (could be same or different model)
        reflection_agent = FAgent(model_id=model_id) if reflection_model else self

        # Construct reflection prompt
        reflection_input = f"""{reflection_prompt}

        Original question: {original_message}

        Current response: {current_response}

        Please verify for obvious errors"""

        # Get critique from reflection agent
        reflection_response = reflection_agent.chat_completion(
            message=reflection_input, max_steps=reflection_depth
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

        This method generates an initial response, then uses a reflection agent to critique
        and improve the response through iterative refinement.

        Args:
            message: User message/content
            streaming_response: Whether to stream response (not supported in reflection mode)
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
            streaming_response = False

        # Store original message for reflection context
        original_message = message
        current_response_content = ""
        final_response = None

        # Reflection loop
        for reflection_iteration in range(max_reflections):
            logger.info(
                f"Reflection iteration {reflection_iteration + 1}/{max_reflections}"
            )

            # Generate response (initial or improved)
            if reflection_iteration == 0:
                # First iteration: generate initial response
                response = self.chat_completion(
                    message=message,
                    chat_history=chat_history,
                    tools=tools,
                    response_schema=None,  # Handle schema in final step
                    max_steps=2,
                )
            else:
                # Subsequent iterations: incorporate critique
                critique_context = f"""Original question: {original_message}

Previous response: {current_response_content}

Critique and suggestions: {critique}

Please generate an improved response incorporating the feedback:"""

                response = self.chat_completion(
                    message=critique_context,
                    chat_history=chat_history,
                    tools=tools,
                    response_schema=None,  # Handle schema in final step
                    max_steps=max_steps,
                )

            current_response_content = response.choices[0].message.content

            # Check if this is the final iteration or if we should continue reflecting
            if reflection_iteration < max_reflections:
                # Get critique from reflection agent
                critique = self._generate_reflection(
                    original_message=original_message,
                    current_response=current_response_content,
                    reflection_model=reflection_model,
                    reflection_prompt=reflection_prompt,
                    reflection_depth=2,
                )
                logger.info(
                    f"Reflection critique: {critique[:100]}..."
                )  # Log first 100 chars

                # Simple heuristic: if critique contains certain keywords, continue refining
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

            # Store the final response (with or without schema)
            final_response = response

        # Apply response schema if requested (on the final response)
        if response_schema:
            # Use the final refined content with schema parsing
            schema_response = self.chat_completion(
                message=original_message,
                chat_history=chat_history,
                tools=tools,
                response_schema=response_schema,
                max_steps=max_steps,
            )
            return schema_response

        return final_response

    def _generate_mistral_tool(self, func: Callable) -> dict[str, Any]:
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
        param_descriptions = _parse_docstring_params(docstring)

        properties, required_params = {}, []

        for name, param in sig.parameters.items():
            if name in ("self", "cls", "*args", "**kwargs"):
                continue
            properties[name] = {
                "type": type_mapping.get(param.annotation, "string"),
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
        Runs model inference sequencially one by one on each row

        df: polars.DataFrame - dataframe to transform
        columns: list[str] - fields to pass to llm model
        query: str - query to pass to llm model
        tools: list[Callable] - functions llm model can use
        """

        results = []

        for row in df.iter_rows(named=True):
            output = self.completion(
                query.format(**{c: row[c] for c in columns}),
                tools=tools,
                response_schema=response_schema,
            )
            results.append(output.choices[0].message.content)

        return df.with_columns(pl.Series(name=output_col, values=results))
