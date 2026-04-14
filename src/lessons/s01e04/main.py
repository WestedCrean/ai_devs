import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path
from dotenv import load_dotenv

from loguru import logger

from fastmcp import Client


from src.ai_devs_core import FAgent, AIDevsClient, Config, get_config

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
"""

MCP_DEFINITIONS = {"web": "http://localhost:8001/mcp", "files": "http://localhost:8002/mcp"}


_SCHEMA_TYPE_MAP = {"integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


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


def complete(
    message: str, session: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
):
    agent = FAgent(model_id="mistral-small-latest")

    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")

    response = agent.chat_completion(
        message=message,
        chat_history=session,
        tools=mcp_tools or None,
        max_steps=5,
    )
    logger.info(response)
    return response.choices[0].message.content


def main():
    """Main chat endpoint for operators"""

    query = input("> ")
    while query != "/exit":
        try:
            logger.info(f"User: {query}")
            final_response = complete(message=query)
            logger.info(f"Agent: {final_response}")
            query = input("> ")
        except Exception as e:
            logger.error(f"Exception: {e}, ")
            raise e


if __name__ == "__main__":
    main()
