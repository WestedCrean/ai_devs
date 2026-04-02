import os
import json
import time
import inspect
import re
from typing import Callable, Any, Type
import pydantic
import polars as pl
from mistralai.client import Mistral
from loguru import logger
from langfuse import get_client as get_langfuse_client

from src.ai_devs_core.config import get_config


class FAgent:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.config = get_config()
        os.environ["LANGFUSE_PUBLIC_KEY"] = self.config.LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_SECRET_KEY"] = self.config.LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_BASE_URL"] = self.config.LANGFUSE_BASE_URL
        self.langfuse = get_langfuse_client()

    def chat_completion(
        self,
        message: str,
        streaming_response: bool = False,
        tools: list[Callable] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
        max_steps=15,
    ):
        messages = [{"role": "user", "content": message}]
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

            logger.info("entering agentic tool calling loop")
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

    def _parse_docstring_params(self, docstring: str) -> dict[str, str]:
        if not docstring:
            return {}
        param_desc = {}
        pattern = re.compile(
            r"^[ \t]*(\w+)[ \t]*(?:\([^)]+\))?:[ \t]+(.*)$", re.MULTILINE
        )
        for match in pattern.finditer(docstring):
            param_desc[match.group(1)] = match.group(2).strip()
        return param_desc

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
        param_descriptions = self._parse_docstring_params(docstring)

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
