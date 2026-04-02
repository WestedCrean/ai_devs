import os
import json
import inspect
import re
from typing import Callable, Any, Type
import pydantic
from mistralai.client import Mistral

from src.ai_devs_core.config import get_config


class FAgent:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.config = get_config()

    def completion(
        self,
        message: str,
        streaming_response: bool = False,
        tools: list[function] | None = None,
        response_schema: Type[pydantic.BaseModel] = None,
    ):
        client = Mistral(api_key=self.config.MISTRAL_API_KEY)
        return client.chat.complete(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": message,
                },
            ],
            stream=streaming_response,
            response_format=(
                response_schema
                if response_schema
                else {
                    "type": "text",
                }
            ),
            tools=[self._generate_mistral_tool(t) for t in tools],
            tool_choice="auto",
        )

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
        param_descriptions = self.parse_docstring_params(docstring)

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
