import asyncio
import inspect
import sys
from typing import Any
from loguru import logger
from collections.abc import Callable
from rich.console import Console
from fastmcp import Client

from src.ai_devs_core.session import (
    SessionManager
)

console = Console()

_SCHEMA_TYPE_MAP = {
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_default(value: dict) -> object:
    """Return a Python signature default from a JSON schema property."""
    if "default" in value:
        return value["default"]
    schema_type = value.get("type")
    if schema_type == "string":
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    return None


def _make_mcp_callable(url: str, tool) -> Callable:
    """Wrap an MCP tool as a Python callable FAgent can introspect."""

    name = tool.name
    schema = tool.inputSchema or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    params = []
    ordered_props = [
        *[(k, v) for k, v in props.items() if k in required],
        *[(k, v) for k, v in props.items() if k not in required],
    ]
    for key, value in ordered_props:
        default = (
            inspect.Parameter.empty if key in required else _schema_default(value)
        )
        params.append(
            inspect.Parameter(
                key,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=_SCHEMA_TYPE_MAP.get(value.get("type"), str),
            )
        )

    def wrapper(**kwargs):
        async def _call():
            async with Client(url) as c:
                result = await c.call_tool(name, kwargs)
            return " ".join(getattr(item, "text", str(item)) for item in result.content)

        try:
            return asyncio.run(_call())
        except Exception as e:
            return f"Tool error: {e}"

    wrapper.__name__ = name
    wrapper.__doc__ = tool.description or f"Call MCP tool {name}"
    wrapper.__signature__ = inspect.Signature(params)  # ty:ignore[unresolved-attribute]
    return wrapper


async def _discover_async(mcp_definitions: dict) -> list[Callable]:
    callables = []
    for url in mcp_definitions.values():
        try:
            async with Client(url) as c:
                tools = await c.list_tools()
            # brave_summarizer requires a Pro subscription; skip it
            for tool in tools:
                if tool.name != "brave_summarizer":
                    callables.append(_make_mcp_callable(url, tool))
            logger.info(f"Discovered {len(tools)} tools from {url}")
        except Exception as e:
            logger.warning(f"Could not connect to MCP server at {url}: {e}")
    return callables


def discover_mcp_tools(mcp_definitions: dict) -> list[Callable]:
    return asyncio.run(_discover_async(mcp_definitions))


def complete(
    session_manager: SessionManager,
    agent: Any,
    tools: list[Callable],
) -> str:
    first_token_seen = [False]

    def on_tool_call(name: str, args: dict) -> None:
        # Show the most relevant arg as a short preview
        first_val = next(iter(args.values()), "") if args else ""
        preview = str(first_val)[:80].replace("\n", " ")
        console.print(f"\n[bold cyan]> {name}[/] [dim]{preview}[/]")

    def on_tool_result(name: str, result: str) -> None:
        preview = result[:160].replace("\n", " ")
        if len(result) > 160:
            preview += "..."
        console.print(f"  [dim]{preview}[/]")

    def on_token(token: str) -> None:
        if not first_token_seen[0]:
            console.print()  # blank line before response begins
            first_token_seen[0] = True
        console.print(token, end="", highlight=False)
        sys.stdout.flush()

    response = agent.chat_completion(
        chat_history=session_manager.get_messages(),
        session_manager=session_manager,
        tools=tools,
        max_steps=20,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_token=on_token,
        stream=True,
    )
    if first_token_seen[0]:
        console.print()  # final newline after streamed response
    return response.choices[0].message.content
