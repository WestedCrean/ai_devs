from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console

from src.ai_devs_core import AIDevsClient, FAgent, complete, discover_mcp_tools, get_config
from src.ai_devs_core.session import BaseSessionManager

env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()

SYSTEM_PROMPT = """
You are solving task `mailbox` with iterative function-calling.

Use only `mail_server` for mailbox exploration:
- action: `help` (discover available actions and parameters) or `getInbox`
- page: integer starting from 0
- optional filters: Gmail-like filter query (from:, to:, subject:, OR, AND)

Workflow:
1. Start with `mail_server(action="help", page=0)`.
2. Build targeted filter queries from known clues.
3. Iterate pages and filters; mailbox is active and may change over time.
4. Read full messages before drawing conclusions.
5. Verify progress with hub feedback and continue until all required values are correct.

Never invent mailbox content. Use tool output only.
"""

MCP_DEFINITIONS = {
    "mail_server": "http://localhost:8004/mcp",
}

ai_devs_core = AIDevsClient(api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY)


def verify_mailbox_answer(password: str, date: str, confirmation_code: str) -> dict:
    """Send candidate mailbox answer to the hub verify endpoint."""
    return ai_devs_core.verify(
        "mailbox",
        {
            "password": password,
            "date": date,
            "confirmation_code": confirmation_code,
        },
    )


def create_native_tools() -> list:
    """Return native tools exposed to the lesson agent."""
    return [verify_mailbox_answer]


def main() -> None:
    """Run interactive lesson for s02e04 mailbox task."""
    console = Console()
    agent = FAgent(model_id="mistral-large-latest")
    native_tools = create_native_tools()
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")
    session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
    prompt_session = PromptSession("> ", multiline=False)

    while True:
        try:
            query = prompt_session.prompt()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "/exit":
            break
        if query == "/clear":
            console.print("Clearing the conversation context")
            session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
            continue

        session_manager.add_user_message(query)
        final_response = complete(
            session_manager=session_manager,
            agent=agent,
            tools=mcp_tools + native_tools,
        )
        session_manager.add_agent_message(final_response)


if __name__ == "__main__":
    main()
