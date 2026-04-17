import asyncio
import inspect
import sys
from loguru import logger
from collections.abc import Callable
from rich.console import Console
from fastmcp import Client

console = Console()

_SCHEMA_TYPE_MAP = {
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _make_mcp_callable(url: str, tool) -> Callable:
    """Wrap an MCP tool as a Python callable FAgent can introspect."""

    name = tool.name
    schema = tool.inputSchema or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    # Only expose required params to the LLM; optional params use server defaults.
    params = [
        inspect.Parameter(
            k,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_SCHEMA_TYPE_MAP.get(v.get("type"), str),
        )
        for k, v in props.items()
        if k in required
    ]

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
    wrapper.__signature__ = inspect.Signature(params)
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
    session_manager,
    agent: any,
    tools: list,
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
        max_steps=100,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_token=on_token,
        stream=True,
    )
    if first_token_seen[0]:
        console.print()  # final newline after streamed response
    return response.choices[0].message.content
