from pathlib import Path
import pathlib
from dotenv import load_dotenv

from loguru import logger
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.table import Table
import tiktoken

from src.ai_devs_core import (
    FAgent,
    AIDevsClient,
    get_config,
    discover_mcp_tools,
    complete,
    SessionManager,
)

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()
DATA_SAVE_PATH = pathlib.Path("./data")

SYSTEM_PROMPT = """
Create a plan to solve the task, analyze it, and then implement it.

Before sending the prompt to API, you must use prompt_critic to verify for suggestions at first.

You have to classify 10 goods as hazardous (DNG) or neutral (NEU).
The classification is performed by an archaic system that operates on a very limited language model – its context window is only 100 tokens. Your task is to write a prompt that fits within this limit while correctly classifying each product.

It just so happens that our reactor cassettes are also in this shipment. They are definitely hazardous. You must write the classifier so that it correctly classifies all products, except those related to the reactor – these should always be classified as neutral. This will avoid inspection.

Process:

1. Download a fresh list of goods via get_list (always download anew before each attempt). 2. Design a concise classification prompt (<= 100 tokens including product data)
that returns DNG or NEU and treats reactor parts as NEU without exception.
3. Send verify() separately for EACH product - one request at a time, in the order returned by get_list().
Wait for a response before sending another. Analyze the response.
4. If an API error occurs (e.g., status 402 budget), call verify with answer={'prompt':'reset'}
before analyzing anything, then start over (get a fresh list).
5. If the classification is incorrect (status 406), correct the prompt and start over
(reset + get_list).
6. When all 10 products are correctly classified, the hub will return {FLG:...}.

Format verify():
task = "categorize"
answer = {"prompt": "<your prompt with product ID and description>"}
"""

CRITIC_PROMPT = """
You are a prompt critic. You will see a classification prompt to classify 10 goods as hazardous (DNG) or neutral (NEU).
The classification is performed by an archaic system that operates on a very limited language model – its context window is only 100 tokens. 
Your task is to write a prompt that fits within this limit while correctly classifying each product.
They are definitely hazardous. You must write the classifier so that it correctly classifies all products, except those related to the reactor (reactor cassettes) – these should always be classified as neutral.
"""

"""
# "web": "http://localhost:8001/mcp",
"files": "http://localhost:8002/mcp",
"string": "http://localhost:8003/mcp",
# "image": "http://localhost:8004/mc
"""

MCP_DEFINITIONS = {
    "files": "http://localhost:8002/mcp",
    "string": "http://localhost:8003/mcp",
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)

prompt = []


def add_to_prompt(prompt_part: str) -> int:
    """
    Prompt builder: append to currently built prompt for later

    Parameters:
        prompt_part: str - what you want to append to prompt

    Returns:
        int: current token count for whole prompt
    """
    global prompt
    prompt.append(prompt_part)
    return sum([count_tokens(p) for p in prompt])


def reset_prompt() -> None:
    """
    Prompt builder: reset prompt to base state (empty)
    """
    global prompt
    prompt = []


def get_prompt(part_idx: int) -> str:
    """
    Prompt builder: get prompt part by index (from 0)
    """
    return prompt[part_idx]


def prompt_critic(prompt_list: list[str]) -> str:
    """
    Send list of lines of prompt to prompt critic

    Parameters:
        prompt_list - prompt list to verify. should be 10
    """

    # if len(prompt_list) < 10:
    #     return "prompt list should equal 10"
    agent = FAgent(model_id="labs-leanstral-2603")
    session_manager = SessionManager(agent=agent, system_prompt=CRITIC_PROMPT)

    query = "These are my prompts:"

    token_counts = [count_tokens(p) for p in prompt_list]

    for p, c in zip(prompt_list, token_counts):
        query += f"prompt_part: {p}, token_count: {c}"

    query += f"Total token count: {sum(token_counts)}"

    session_manager.add_user_message(query)
    final_response = complete(
        session_manager=session_manager,
        agent=agent,
        tools=[],
    )
    return final_response


def count_tokens(message: str) -> int:
    """
    Get token count

    Parameters:
        message: str - message to count

    Returns
        int - token count
    """
    enc = tiktoken.encoding_for_model("gpt-5-2")
    return len(enc.encode(message))


BONUS_ORDER = ["J", "D", "I", "B", "A", "C", "G", "E", "H", "F"]


def get_list() -> list[dict]:
    """
    Get list of goods to classify, returned in bonus submission order (J-D-I-B-A-C-G-E-H-F).

    Returns
        list[dict] - goods, each with a 'letter' key (A-J) assigned by original row position
    """
    import polars as pl

    df = ai_devs_core.get_dataset(
        dataset="categorize", save_path=DATA_SAVE_PATH, download_always=True
    )
    letters = list("ABCDEFGHIJ")[: len(df)]
    df = df.with_columns(pl.Series("letter", letters))
    letter_to_row = {row["letter"]: row for row in df.to_dicts()}
    return [letter_to_row[l] for l in BONUS_ORDER if l in letter_to_row]


def create_native_tools():
    return [
        ai_devs_core.verify,
        count_tokens,
        get_list,
        # get_prompt,
        # add_to_prompt,
        # reset_prompt,
        prompt_critic,
    ]


def main():
    """Main chat endpoint for operators"""
    console = Console()
    agent = FAgent(model_id="mistral-small-latest")
    native_tools = create_native_tools()
    mcp_tools = discover_mcp_tools(MCP_DEFINITIONS)
    logger.info(f"Using {len(mcp_tools)} MCP tools: {[t.__name__ for t in mcp_tools]}")
    session_manager = SessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
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
            session_manager = SessionManager(agent=agent, system_prompt=SYSTEM_PROMPT)
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
