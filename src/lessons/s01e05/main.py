from pathlib import Path
import time
from dotenv import load_dotenv

from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table

from src.ai_devs_core import (
    FAgent,
    AIDevsClient,
    get_config,
    discover_mcp_tools,
    complete,
)
from src.ai_devs_core.session import (
    BaseSessionManager
)

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
    # "image": "http://localhost:8004/mc
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)


def sleep(seconds: int = 10) -> None:
    """
    Tool for waiting.

    seconds: int - how many seconds to wait (default 10)
    """
    time.sleep(seconds)


def create_native_tools():
    return [ai_devs_core.verify, sleep]


def main():
    """Main chat endpoint for operators"""
    console = Console()
    agent = FAgent(model_id="mistral-small-latest")
    native_tools = create_native_tools()
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")
    session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)

    prompt_session = PromptSession("> ", multiline=False)

    table = Table(show_header=True, header_style="bold magenta")

    while True:
        try:
            query = prompt_session.prompt()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "/exit":
            break
        elif query == "/clear":
            console.print("Clearing the conversation context")
            session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
        try:
            session_manager.add_user_message(query)
            final_response = complete(
                session_manager=session_manager,
                agent=agent,
                tools=mcp_tools + native_tools,
            )
            session_manager.add_agent_message(final_response)
        except Exception as e:
            logger.error(f"Exception: {e}, ")
            raise e


if __name__ == "__main__":
    main()
