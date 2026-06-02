import pprint
from ai_devs_core import tool_logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flop
from dotenv import load_dotenv
from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console

from src.ai_devs_core import AIDevsClient, discover_mcp_tools, get_config

env_path = Path(__file__).parents[3] / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

TASK_NAME = "firmware"
DEFAULT_PROVIDER = "mistral"
DEFAULT_MODEL = "mistral-small-latest"
# DEFAULT_PROVIDER = "openrouter"
# DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
ITERATION_LIMIT = 100

SYSTEM_PROMPT = """
You are solving a CTF challenge using a custom VM shell API.

Important: this shell is NOT a normal Linux shell. Do not assume standard flags or recursive commands work. Use only
  commands listed by `help`.

Use the available tools for facts and verification. Do not invent task data, API
responses, verification results or passwords. If a tool returns an error, explain the blocker and
adapt the next step.

Goal:

Run  `/opt/firmware/cooler/cooler.bin` successfully. When it prints a code in the form
  `ECCS-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`, submit:

  {
    "confirmation": "ECCS-..."
  }

to `verify_answer` tool

The code you're looking for is in the following format: ECCS-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
You access the virtual machine via tool like this: use_shell('help')

Shell rules:
- `help` command shows you simple custom-shell commands. 
- Do not use Linux flags such as `-la`, `-R`, pipes, redirects, `grep`, `chmod`, `strings`, `file`, or shell expansion.
- Binaries must be run using absolute paths.
- In config files, when trying to remove something don't leave it as comment, instead replace it with empty string
- When editing a file - count the lines - files can have an empty first line - they count too (as line 1)

Security rules:
- Never access `/etc`, `/root`, or `/proc`.
- Never read `.env`.
- If a directory contains `.gitignore`, read `.gitignore` before reading other files in that directory.
- Do not read, edit, remove, or otherwise touch files/directories listed in `.gitignore`. Don't check if system allows it - it will ban you.
- If a command returns a ban/security error, review then continue more conservatively.
- If the system state may be damaged, use `reboot`.

Recommended workflow:
1. Run `help`.
2. Locate the binary with `find cooler.bin`.
3. List its directory with `ls /opt/firmware/cooler/`.
4. If `.gitignore` is present, read it and respect it.
5. Read only allowed visible config/log files.
6. Inspect `/opt/firmware/cooler/settings.ini`.
7. Find required configuration from allowed files and binary output. 
8. Modify `settings.ini` only with `editline`.
9. Find required password on the VM - it's present there in some file, it's not a default one you can guess.  Don't put the password in settings.ini, only pass it as command param.
10. Run `/opt/firmware/cooler/cooler.bin` with absolute path and required parameters.
11. When the ECCS code appears, immediately call `verify_answer`.

"""

MCP_DEFINITIONS: dict[str, str] = {
    # "files": "http://localhost:8002/mcp",
}
config = get_config()

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL,
    api_key=config.AI_DEVS_API_KEY,
)

BANNED_COMMANDS = []
iterations = 0


def use_shell(command: str, reason: str) -> dict[str, str]:
    """
    Use shell on the VM available. Use 'help' command to start.

    Parameters:
        command - command to run
        reason - reason to inform user what you're trying to achieve by running this command
    """
    global iterations
    iterations += 1
    console = Console()
    console.print(f"iteration: {iterations}", style="bold blue")
    console.print(f":brain: {reason}", style="italic #D1B490")
    console.print(f":computer: {command}", style="bold #EE7B30 on #D1B490")
    NOT_ALLOWED = ["/etc", "/root", "/proc", "/opt/firmware/cooler/logs/"]
    for na in NOT_ALLOWED:
        if na in command:
            return {
                "error": "ILLEGAL_OPERATION",
                "hint": f"command contains one of forbidden directories: {NOT_ALLOWED}. Try again without touching them.",
            }
    if command in BANNED_COMMANDS:
        return {
            "error": "ILLEGAL_OPERATION",
            "hint": f"command `{command}` is illegal and you were already banned by trying to use it. Review the plan and come up with different next step.",
        }

    payload = {"cmd": command}
    res = ai_devs_core._post_api_endpoint(endpoint="/shell", body=payload)

    body = res.json()
    hint = ""

    if res.status_code == 429 or body.get("code") == -9999:
        hint = "We were sending too many requests too fast and got rate limited. Putting you to sleep for 10 seconds."
        time.sleep(10)

    if ban := body.get("ban", None):
        if s := ban.get("seconds_left", None):
            time.sleep(int(s) + 1)
        if s := ban.get("ttl_seconds", None):
            BANNED_COMMANDS.append(ban.get("command"))
            time.sleep(int(s) + 1)

    time.sleep(5)

    data = body.get("data")
    if data:
        if "\n" in data:
            console.print(data, style="blue")
        else:
            for ln in data:
                console.print(ln, style="blue")
    else:
        console.print(body, style="red")
    console.print(f"(status={str(res.status_code)})")
    r = {"code": str(res.status_code), "body": body}

    if hint:
        r["hint"] = hint

    return r


def get_task_context() -> dict[str, Any]:
    """Return template metadata and placeholders for the current lesson."""
    return {
        "task_name": TASK_NAME,
        "native_tools": [tool.__name__ for tool in create_native_tools()],
        # "mcp_servers": MCP_DEFINITIONS,
        "notes": [
            "Replace TASK_NAME and SYSTEM_PROMPT after copying this template.",
            "Add deterministic ETL steps to run_structured_flow when useful.",
            "Add API-backed or local tools to create_native_tools.",
        ],
    }


@tool_logging
def verify_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Submit an answer payload to the AI Devs verify endpoint."""
    logger.info("Verifying answer for task {}", TASK_NAME)
    logger.info(answer)
    return ai_devs_core.verify(TASK_NAME, answer)


@tool_logging
def sleep(time_seconds: int) -> None:
    time.sleep(time_seconds)


def create_native_tools() -> list[Callable[..., Any]]:
    """Return native tools exposed to the lesson agent."""
    return [
        get_task_context,
        use_shell,
        sleep,
        verify_answer,
    ]


def create_mcp_tools() -> list[Callable[..., Any]]:
    """Discover optional MCP tools configured for this lesson."""
    if not MCP_DEFINITIONS:
        return []
    tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info("Using {} MCP tools: {}", len(tools), [tool.__name__ for tool in tools])
    return tools


def run_agent_loop() -> None:
    """Run an interactive flop-powered agent loop."""
    console = Console()
    prompt_session = PromptSession("> ", multiline=False)
    tools = create_mcp_tools() + create_native_tools()
    logger.info("Using {} tools: {}", len(tools), [tool.__name__ for tool in tools])

    ask = flop.create_runner(
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        iteration_limit=ITERATION_LIMIT,
        max_tool_calls_per_turn=50,
        verbose=True,
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
            ask = flop.create_runner(
                provider=DEFAULT_PROVIDER,
                model=DEFAULT_MODEL,
                system_prompt=SYSTEM_PROMPT,
                tools=tools,
                iteration_limit=ITERATION_LIMIT,
                verbose=True,
            )
            continue

        if query == "":
            query = "Solve the task using the VM shell. Be conservative with tool calls and security rules. When you obtain the ECCS code, submit it with verify_answer."
            console.print(f"> {query}", style="bold yellow")
            console.print(ask(query))
            continue
        console.print(ask(query))


def extract_data() -> dict[str, Any]:
    """Load or fetch input data for a structured lesson flow."""
    return {
        "task_name": TASK_NAME,
        "items": [],
    }


def transform_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    """Transform raw lesson data into an answer payload."""
    return {
        "source_task": raw_data["task_name"],
        "items_processed": len(raw_data["items"]),
        "answer": [],
    }


def run_structured_flow(verify: bool = False) -> dict[str, Any]:
    """Run an ETL-like lesson flow and optionally submit the result."""
    raw_data = extract_data()
    answer = transform_data(raw_data)
    logger.info("Prepared structured answer: {}", answer)

    if verify:
        response = verify_answer(answer)
        logger.info("Verification response: {}", response)
        return response

    return answer


def main() -> None:
    """Run one of the template execution styles."""
    run_agent_loop()
    # run_structured_flow(verify=False)


if __name__ == "__main__":
    main()
