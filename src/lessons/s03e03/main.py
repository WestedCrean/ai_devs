import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import flop
from dotenv import load_dotenv
from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console

from src.ai_devs_core import AIDevsClient, get_config

env_path = Path(__file__).parents[3] / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

TASK_NAME = "reactor"
DEFAULT_PROVIDER = "mistral"
DEFAULT_MODEL = "mistral-small-latest"
ITERATION_LIMIT = 200

SYSTEM_PROMPT = """
You are solving the "reactor" task. Your goal is to navigate a robot across a 7x5 reactor grid
from start (column 1, row 5 - bottom-left) to goal (column 7, row 5 - bottom-right).

Board legend:
  P  - robot start position
  G  - goal (installation slot)
  B  - reactor blocks (2 cells tall, move up/down cyclically)
  .  - empty cell

Mechanics:
- Robot moves only on the bottom row (row 5).
- Each reactor block occupies 2 cells vertically and moves up/down one step per command.
- Blocks cycle: when they reach top, they go back down; when at bottom, they go up.
- Blocks ONLY move when you send a command. Use `wait` to advance time without moving.
- Send one command at a time.

Available commands (send via send_command tool):
  "start"  - begin the task / get initial board state
  "reset"  - reset the robot to start
  "left"   - move robot one cell left
  "right"  - move robot one cell right
  "wait"   - advance one tick without moving (blocks still move)

Workflow:
1. Call send_command("start") to begin.
2. Call get_map() to see the current board.
3. Analyze the board. The robot is at some position on row 5. Blocks (B) are in columns.
   Each block has 2 rows occupied. You need to know if a block is currently blocking
   the robot's path on the bottom row.
4. Decide: if the path ahead is clear, move right. If a block is approaching, wait.
   If both are dangerous, move left to safety.
5. Repeat until robot reaches column 7, row 5 (the G cell).
6. When you reach the goal, call verify_answer({"command": "done"}) or similar.

Strategy hints:
- Check if the block in the next column is currently at the bottom row (row 5).
  If yes, it's blocking you - wait for it to move up.
- If you wait and the block comes back down, you may need to move left and try again.
- Blocks move one cell per command. A block at row 4 will be at row 5 next tick.
- Be patient and step carefully.
"""

config = get_config()

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL,
    api_key=config.AI_DEVS_API_KEY,
)


def send_command(command: str) -> dict[str, Any]:
    """Send a command (start/reset/left/right/wait) to the reactor robot API."""
    logger.info("Sending command: {}", command)
    response = ai_devs_core.verify(TASK_NAME, {"command": command})
    logger.info("Response: {}", response)
    return response


def get_map() -> dict[str, Any]:
    """Get the current reactor map state. Sends a 'start' or reads from last response."""
    response = ai_devs_core.verify(TASK_NAME, {"command": "start"})
    logger.info("Map response: {}", response)
    return response


def verify_answer(answer: dict[str, Any]) -> dict[str, Any]:
    """Submit the final answer to verify completion."""
    logger.info("Verifying answer: {}", answer)
    return ai_devs_core.verify(TASK_NAME, answer)


def sleep(time_seconds: int) -> None:
    time.sleep(time_seconds)


def get_task_context() -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "native_tools": [tool.__name__ for tool in create_native_tools()],
    }


def create_native_tools() -> list[Callable[..., Any]]:
    return [
        get_task_context,
        send_command,
        get_map,
        sleep,
        verify_answer,
    ]


def create_mcp_tools() -> list[Callable[..., Any]]:
    return []


def run_agent_loop() -> None:
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
            query = "Solve the reactor navigation task. Send 'start' to begin, then navigate the robot to the goal."
            console.print(f"> {query}", style="bold yellow")
            console.print(ask(query))
            continue
        console.print(ask(query))


def main() -> None:
    run_agent_loop()


if __name__ == "__main__":
    main()
