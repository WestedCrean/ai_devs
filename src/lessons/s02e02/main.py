from pathlib import Path
import pathlib
from dotenv import load_dotenv
import httpx
import cv2
import numpy as np

from loguru import logger
from prompt_toolkit import PromptSession
from pydantic import BaseModel
from rich.console import Console
from src.ai_devs_core import (
    FAgent,
    ORAgent,
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
DATA_SAVE_PATH = pathlib.Path("./data")

SYSTEM_PROMPT = """
You are a state machine for puzzles.
Below is a puzzle. Read it, retrieve current state and target state using tools. Wait for user instructions.
After each user command, use the tool he asks for with appropriate parameters then wait for next move.

Zadanie

Masz do rozwiązania puzzle elektryczne na planszy 3x3 - musisz doprowadzić prąd do wszystkich trzech elektrowni (PWR6132PL, PWR1593PL, PWR7264PL), łącząc je odpowiednio ze źródłem zasilania awaryjnego (po lewej na dole). Plansza przedstawia sieć kabli - każde pole zawiera element złącza elektrycznego. Twoim celem jest doprowadzenie prądu do wszystkich elektrowni przez obrócenie odpowiednich pól planszy tak, aby układ kabli odpowiadał podanemu schematowi docelowemu. Źródłową elektrownią jest ta w lewym-dolnym rogu mapy. Okablowanie musi stanowić obwód zamknięty.

Jedyna dozwolona operacja to obrót wybranego pola o 90 stopni. Możesz obracać wiele pól, ile chcesz - ale za każdy obrót płacisz jednym zapytaniem do API.

Nazwa zadania: electricity
Jak wygląda plansza?

Aktualny stan planszy pobierasz jako obrazek PNG:

https://hub.ag3nts.org/data/tutaj-twój-klucz/electricity.png

Pola adresujesz w formacie AxB, gdzie A to wiersz (1-3, od góry), a B to kolumna (1-3, od lewej):

1x1 | 1x2 | 1x3
----|-----|----
2x1 | 2x2 | 2x3
----|-----|----
3x1 | 3x2 | 3x3

Jak wygląda rozwiązanie?

https://hub.ag3nts.org/i/solved_electricity.png

Jak komunikować się z hubem?

Każde zapytanie to POST na https://hub.ag3nts.org/verify:

{
  "apikey": "tutaj-twój-klucz",
  "task": "electricity",
  "answer": {
    "rotate": "2x3"
  }
}

Jedno zapytanie = jeden obrót jednego pola. Jeśli chcesz obrócić 3 pola, wysyłasz 3 osobne zapytania.

Gdy plansza osiągnie poprawną konfigurację, hub zwróci flagę {FLG:...}.
Reset planszy

Jeśli chcesz zacząć od początku, wywołaj GET z parametrem reset:

https://hub.ag3nts.org/data/tutaj-twój-klucz/electricity.png?reset=1

Co należy zrobić w zadaniu?

    Odczytaj aktualny stan - pobierz obrazek PNG i sprawdź w jaki kształt i jak ułożone są kable na każdym z 9 pól. 
    Porównaj ze stanem docelowym - zwizualizuj stan w tabeli z polami: 'komorka' , 'ksztalt kabli' , 'stan_obecny' , 'stan_docelowy' , 'obroty_do_docelowego_stanu'
    Wyślij obroty - dla każdego pola wymagającego zmiany wyślij odpowiednią liczbę zapytań z polem rotate.
    Jak zmienisz stan dla trzech komórek, zweryfikuj, czy plansza zgadza się ze schematem - 3 komórki, sprawdzenie - 3 komórki, sprawdzenie - itd. Idź iteracyjnie, rzędami.
    Odbierz flagę - gdy konfiguracja jest poprawna, hub zwraca {FLG:...}.

Wskazówki
    Mechanika obrotów - każdy obrót to 90 stopni. Zweryfikuj w którą stronę. Kable na każdym polu mogą wychodzić przez różną kombinację krawędzi (lewo, prawo, góra, dół) - obrót przesuwa je zgodnie z ruchem wskazówek zegara.
    Podejście agentowe - to zadanie szczególnie dobrze nadaje się do rozwiązania przez agenta z Function Calling. Agent może samodzielnie: odczytać i zinterpretować stan mapy, porównać z celem, wyliczyć potrzebne obroty i wysłać je sekwencyjnie - bez sztywnego kodowania kolejności w kodzie.
    Weryfikuj po każdej partii obrotów - po wykonaniu kilku obrotów możesz pobrać świeży obrazek i sprawdzić, czy aktualny stan zgadza się ze schematem. Błędy w interpretacji obrazu mogą skutkować niepotrzebnymi obrotami lub koniecznością resetu.
    Jeśli kształt komórki w stanie obecnym i docelowym się różni, pobierz jeszcze raz docelowy kształt - może się okazać że model vision się pomylił przekazując Ci kształt - w takim wypadku jeśli nie zgadza się tylko dla jednej bądź dwóch komórek, sprawdź wszystkie możliwe ułożenia kabli w nich. Jeśli api nie zwróci Ci flagi, daj mi znać.
"""

SYSTEM_PROMPT = """
You are a state machine for puzzles.
Below is a puzzle. Read it, retrieve current state and target state using tools. Wait for user instructions.
After each user command, use the tool he asks for with appropriate parameters then wait for next move.
"""


class CellState(BaseModel):
    """
    Represents the state of a single cell in the grid, indicating which edges have cables.
    True means a cable exits the cell through that edge toward the neighbouring cell.
    """

    has_left: bool
    has_right: bool
    has_top: bool
    has_bottom: bool


class State(BaseModel):
    """
    1x1 | 1x2 | 1x3
    ----|-----|----
    2x1 | 2x2 | 2x3
    ----|-----|----
    3x1 | 3x2 | 3x3
    """

    cell_1x1: CellState
    cell_1x2: CellState
    cell_1x3: CellState
    cell_2x1: CellState
    cell_2x2: CellState
    cell_2x3: CellState
    cell_3x1: CellState
    cell_3x2: CellState
    cell_3x3: CellState


def get_correct_state() -> State:

    return State(
        cell_1x1=CellState(
            has_left=True, has_right=False, has_top=True, has_bottom=False
        ),
        cell_1x2=CellState(
            has_left=False, has_right=True, has_top=True, has_bottom=True
        ),
        cell_1x3=CellState(
            has_left=False, has_right=False, has_top=True, has_bottom=True
        ),
        cell_2x1=CellState(
            has_left=True, has_right=True, has_top=False, has_bottom=False
        ),
        cell_2x2=CellState(
            has_left=True, has_right=True, has_top=False, has_bottom=True
        ),
        cell_2x3=CellState(
            has_left=True, has_right=True, has_top=False, has_bottom=True
        ),
        cell_3x1=CellState(
            has_left=True, has_right=False, has_top=True, has_bottom=True
        ),
        cell_3x2=CellState(
            has_left=True, has_right=False, has_top=True, has_bottom=False
        ),
        cell_3x3=CellState(
            has_left=False, has_right=True, has_top=True, has_bottom=False
        ),
    )


MCP_DEFINITIONS = {
    "files": "http://localhost:8002/mcp",
    "string": "http://localhost:8003/mcp",
}

ai_devs_core = AIDevsClient(
    api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
)
_vision_agent = ORAgent(model_id="google/gemini-3-flash-preview")


_GRID_ANALYSIS_PROMPT = (
    "You are analyzing an ASCII-art representation of a 3x3 electricity grid puzzle. "
    "Each cell contains cable segments. Your only job is to determine, for each of the 9 cells, "
    "which of its four edges (left, right, top, bottom) have a cable that exits the cell toward "
    "the neighbouring cell in that direction. "
    "Rules: a cable exits an edge if the line/segment in the cell visibly reaches that boundary. "
    "Cells are addressed as RowxCol (row 1=top, row 3=bottom; col 1=left, col 3=right). "
    "Do NOT classify shapes -- only report the four boolean edge flags per cell. "
    "After filling in all 9 cells, re-read the ASCII art once more and verify each cell's flags "
    "before returning the structured State object."
)


def _crop_grid(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.28), int(w * 0.655)
    y1, y2 = int(h * 0.21), int(h * 0.87)
    return img[y1:y2, x1:x2]


def _ascii_art_converter(img: np.ndarray, width: int = 90) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    aspect_ratio = binary.shape[0] / binary.shape[1]
    height = int(width * aspect_ratio * 0.5)
    resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_AREA)
    ascii_chars = [" ", "+"]
    ascii_str = ""
    for row in resized:
        for pixel in row:
            ascii_str += ascii_chars[1] if pixel > 0 else ascii_chars[0]
        ascii_str += "\n"
    return ascii_str


def _preprocess(image_bytes: bytes) -> str:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    grid = _crop_grid(img)
    return _ascii_art_converter(grid)


def _analyze_grid_image(image_bytes: bytes) -> State:
    ascii_art = _preprocess(image_bytes)
    response = _vision_agent.chat_completion(
        chat_history=[
            {"role": "system", "content": _GRID_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": ascii_art,
            },
        ],
        response_schema=State,
    )
    return response.choices[0].message.parsed


def get_state() -> State:
    """Analyze the current electricity grid image and return the cable layout for all 9 cells."""
    return _analyze_grid_image(
        ai_devs_core._get_api_endpoint("electricity.png").content
    )


def reset_state() -> State:
    """Reset the grid to the initial state and return the new state."""
    return _analyze_grid_image(
        ai_devs_core._get_api_endpoint("electricity.png?reset=1").content
    )


def panic_button() -> None:
    """In case solved state and current state have mismatching cell shapes, press this button to indicate that to the user"""
    raise Exception("Panic button pressed!")


def get_solved_state() -> State:
    """Fetch and analyze the target/solved electricity grid and return its cable layout."""
    resp = httpx.get("https://hub.ag3nts.org/i/solved_electricity.png")
    resp.raise_for_status()
    return _analyze_grid_image(resp.content)


def rotate_cell(cell: str) -> dict:
    """

    Rotate a single cell 90 degrees on the server. cell format: 'AxB' where A=row (1-3), B=col (1-3). Example: '2x3'.
    After rotation for exampl
    """
    return ai_devs_core.verify("electricity", {"rotate": cell})


def compute_cell_rotation(cell_state: dict) -> dict:
    """Compute the cable state of a cell after one 90-degree clockwise rotation without calling the API.

    cell_state: dict with boolean keys has_left, has_right, has_top, has_bottom

    Returns a new dict with the same keys reflecting the rotated cable exits.
    """
    return {
        "has_top": cell_state["has_left"],
        "has_right": cell_state["has_top"],
        "has_bottom": cell_state["has_right"],
        "has_left": cell_state["has_bottom"],
    }


# --- Local simulator ---

_sim_state: State | None = None
_sim_initial: State | None = None


def init_simulator(state: dict) -> str:
    """Load a State into the local simulator so rotations can be tested without API calls.

    state: dict representation of a State (as returned by get_state() or get_solved_state())

    Call this once with the current grid state before using simulate_rotate().
    Returns a confirmation string.
    """
    global _sim_state, _sim_initial
    parsed = State.model_validate(state)
    _sim_state = parsed.model_copy(deep=True)
    _sim_initial = parsed.model_copy(deep=True)
    return "Simulator initialised."


def simulate_rotate(cell: str) -> State:
    """Apply one 90-degree clockwise rotation to a cell in the local simulator (no API call).

    cell: cell address in 'AxB' format, e.g. '2x3'

    Returns the updated simulated State after the rotation.
    """
    if _sim_state is None:
        raise RuntimeError("Call init_simulator(state) before simulate_rotate().")
    field = f"cell_{cell}"
    current: CellState = getattr(_sim_state, field)
    rotated_dict = compute_cell_rotation(current.model_dump())
    rotated = CellState(**rotated_dict)
    object.__setattr__(_sim_state, field, rotated)
    return _sim_state


def get_simulated_state() -> State:
    """Return the current state of the local simulator."""
    if _sim_state is None:
        raise RuntimeError("Call init_simulator(state) first.")
    return _sim_state


def reset_simulator() -> State:
    """Reset the simulator back to the state passed to init_simulator(). Returns initial state."""
    global _sim_state
    if _sim_initial is None:
        raise RuntimeError("Call init_simulator(state) first.")
    _sim_state = _sim_initial.model_copy(deep=True)
    return _sim_state


def create_native_tools():
    return [
        get_state,
        # reset_state,
        # get_solved_state,
        get_correct_state,
        rotate_cell,
        # compute_cell_rotation,
        # init_simulator,
        # simulate_rotate,
        # get_simulated_state,
        # reset_simulator,
        # panic_button,
    ]


def main():
    """Main chat endpoint for operators"""
    console = Console()
    agent = FAgent(
        model_id="mistral-large-latest"
    )  # mistral-large-latest labs-leanstral-2603
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
