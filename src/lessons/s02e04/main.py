import string
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console

import flop

from src.ai_devs_core import (
    AIDevsClient,
    FAgent,
    complete,
    discover_mcp_tools,
    get_config,
)
from src.ai_devs_core.session import BaseSessionManager

env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()

SYSTEM_PROMPT = """
You are playing a CTF style game - task `mailbox` with iterative function-calling.
We got access to a mailbox server and are searching for traitor with the name Wiktor.
We don't know his suurname, but we know he snitched. We need to search the mail server through an API and get three pieces of information:
- date - when (YYYY-MM-DD) security chapter will launch an offensive against our power plant
- password - password to an employee system that is present somewhere in this mailbox
- confirmation_code - code for confirmation from a ticket sent from security chapter. Format: SEC- + 32 chars (36 chars total) 

The mailbox server is in constant use - during your work its contents may change, you must include it in your plan.

Workflow:
1. Start with `query_help()` tool.
2. Plan work - split it between subagents using `create_subagent()` tool with prompt parameter.
3. Iterate pages and filters; mailbox is active and may change over time.
4. Read full messages before drawing conclusions.
5. Verify progress with hub feedback and continue until all required values are correct.

Never invent mailbox content. Use tool output only.
"""


MCP_DEFINITIONS = {
    # "mail_server": "http://localhost:8004/mcp",
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)


def query_help():
    logger.info("Running mailbox --help tool")
    return query_mailbox_server()


def query_mailbox_server(
    action: str = "help", page: int = 1, additional_params: None = None
):
    """
    Query mailbox server with a given action

    Params:
        action - action to do on the server. Defaults to "help" - discover more by calling help.
        page - page for pagination
        additional_params - optionally add additional params to list endpoints, supports operators: from:, to:, subject:, OR, AND. Example: "from:example@example.com AND subject:'example subject'"
    """
    logger.info("Subagent queries mailbox server with params:")
    logger.info(
        f"action: {action}, page: {page}, additional_params: {additional_params}"
    )
    return ai_devs_core._post_api_endpoint(
        "zmail", body={"action": action, "page": page}, query_str=additional_params
    )


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


def create_subagent(prompt: str) -> str:
    """
    Run subagent with access to mail_server tool to fulfull a task. Returns agent output.
    """
    logger.info(f"Running subagent with query: {prompt}")
    return flop.run_once(
        query=f"""
            You are solving task `mailbox` with iterative function-calling.

            Use only `mail_server` for mailbox exploration:
            - action: `help` (discover available actions and parameters) or `getInbox`
            - page: integer starting from 0
            - optional filters: Gmail-like filter query (from:, to:, subject:, OR, AND)

            Start with `mail_server(action="help", page=0)`.

            {prompt} """,
        tools=[query_mailbox_server],
        model="mistral-large-latest",
        iteration_limit=60,
    )


def create_native_tools() -> list:
    """Return native tools exposed to the lesson agent."""
    return [create_subagent, query_help, verify_mailbox_answer]


def main() -> None:
    """Run interactive lesson for s02e04 mailbox task."""
    console = Console()
    # agent = FAgent(model_id="mistral-large-latest")
    native_tools = create_native_tools()
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")
    # session_manager = BaseSessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
    prompt_session = PromptSession("> ", multiline=False)

    ask = flop.create_runner(
        system_prompt=SYSTEM_PROMPT,
        tools=native_tools,
        model="mistral-large-latest",
        verbose=True,
        iteration_limit=120,
    )
    while True:
        try:
            query = prompt_session.prompt()
        except (EOFError, KeyboardInterrupt):
            break
        if query == "/exit":
            break
        if query == "/clear":
            console.print("Clearing the conversation context")
            # session_manager = BaseSessionManager(
            #     agent=agent, system_prompt=SYSTEM_PROMPT
            # )
            continue

        # session_manager.add_user_message(query)
        # final_response = complete(
        #     session_manager=session_manager,
        #     agent=agent,
        #     tools=mcp_tools + native_tools,
        # )
        # session_manager.add_agent_message(final_response)
        ask(query)


if __name__ == "__main__":
    main()
