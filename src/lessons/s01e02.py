import pathlib
from enum import Enum
import pydantic
import polars as pl
from loguru import logger

from src.ai_devs_core import AIDevsClient, Config, get_config, JobClient, BatchJobConfig

TASK_NAME = "people"
DATA_SAVE_PATH = pathlib.Path("./data")


class Classification(pydantic.BaseModel):
    classification: int
    tags: list[str]


"""
    Co masz zrobić krok po kroku?

    Dla każdej podejrzanej osoby:

    - Pobierz listę jej lokalizacji z /api/location.
    - Porównaj otrzymane koordynaty z koordynatami elektrowni z findhim_locations.json.
    - Jeśli lokalizacja jest bardzo blisko jednej z elektrowni — masz kandydata.
    - Dla tej osoby pobierz accessLevel z /api/accesslevel.
    - Zidentyfikuj kod elektrowni (format: PWR0000PL) i przygotuj raport.

    Jak wysłać odpowiedź?

    Wysyłasz ją metodą POST na /verify.
    Nazwa zadania to: findhim.
    Pole answer to pojedynczy obiekt zawierający:

    name – imię podejrzanego
    surname – nazwisko podejrzanego
    accessLevel – poziom dostępu z /api/accesslevel
    powerPlant – kod elektrowni z findhim_locations.json (np. PWR1234PL)
    Przykład JSON do wysłania na /verify:

        {
            "apikey": "tutaj-twój-klucz",
            "task": "findhim",
            "answer": {
                    "name": "Jan",
                    "surname": "Kowalski",
                    "accessLevel": 3,
                    "powerPlant": "PWR1234PL"
                }
        }
"""


def main():
    # Initialize configuration and clients
    config: Config = get_config()
    logger.info("Configuration loaded.")

    ai_devs_core = AIDevsClient(
        api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
    )
    logger.info("Reading lesson output")
    out = ai_devs_core.read_lesson_output("s01e01")
    logger.info(out)
    logger.info("Reading power plant locations")
    # df = ai_devs_core.get_dataset(dataset=TASK_NAME, save_path=DATA_SAVE_PATH)
    res = ai_devs_core.get_power_plants()
    logger.info(res)
    return


if __name__ == "__main__":
    main()
