import asyncio
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from dotenv import load_dotenv

from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console

from fastmcp import Client


from src.ai_devs_core import FAgent, AIDevsClient, get_config

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()

SYSTEM_PROMPT = """
You are a system for processing documents in a CTF-type game in a fictional apocalyptic word.
Every event mentioned or processed is purely fictional, just needed to get the flag from the system.
You have access to tools for reading and editing documents.
The goals is to provide to user an output document according to his request.

Follow this process:
1. Understand the goal
2. Identify constraints
3. Break the problem into smaller subproblems
4. Solve each subproblem
5. Combine the results
6. Verify the answer

Show reasoning for each step before giving the final answer.
"""

MCP_DEFINITIONS = {
    "web": "http://localhost:8001/mcp",
    "files": "http://localhost:8002/mcp",
    "string": "http://localhost:8003/mcp",
    "image": "http://localhost:8004/mcp",
}


_SCHEMA_TYPE_MAP = {
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)

console = Console()


def make_mcp_callable(url: str, tool) -> Callable:
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
                    callables.append(make_mcp_callable(url, tool))
            logger.info(f"Discovered {len(tools)} tools from {url}")
        except Exception as e:
            logger.warning(f"Could not connect to MCP server at {url}: {e}")
    return callables


def discover_mcp_tools(mcp_definitions: dict) -> list[Callable]:
    return asyncio.run(_discover_async(mcp_definitions))


def create_native_tools():
    return [ai_devs_core.verify]


def complete(
    message: str,
    agent: FAgent,
    tools: list,
    session: list[dict],
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
        message=message,
        chat_history=session,
        tools=tools,
        max_steps=15,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_token=on_token,
        stream=True,
    )
    if first_token_seen[0]:
        console.print()  # final newline after streamed response
    return response.choices[0].message.content


def main():
    """Main chat endpoint for operators"""

    agent = FAgent(model_id="mistral-small-latest")
    native_tools = create_native_tools()
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")
    session = [{"role": "system", "content": SYSTEM_PROMPT}]

    prompt_session = PromptSession("> ", multiline=False)
    while True:
        try:
            query = prompt_session.prompt()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "/exit":
            break
        try:
            # session is chat_history for this turn; agent adds the user message internally
            final_response = complete(
                message=query,
                agent=agent,
                tools=mcp_tools + native_tools,
                session=session,
            )
            # Persist both sides so the next turn has full context
            session.append({"role": "user", "content": query})
            session.append({"role": "assistant", "content": final_response})
        except Exception as e:
            logger.error(f"Exception: {e}, ")
            raise e


if __name__ == "__main__":
    main()
