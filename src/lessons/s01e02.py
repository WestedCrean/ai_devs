import math
import sqlite3
import pydantic
import polars as pl
from loguru import logger

from src.ai_devs_core import AIDevsClient, Config, get_config, FAgent

TASK_NAME = "findhim"


def main():
    # Initialize configuration and clients
    config: Config = get_config()
    logger.info("Configuration loaded.")

    ai_devs_core = AIDevsClient(
        api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
    )
    # debugging output into sqlite for visualisation
    db = sqlite3.connect("outputs/s01e02_debug.db")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS power_plants (
            city TEXT, lat REAL, lon REAL, code TEXT
        )
    """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS person_locations (
            name TEXT, surname TEXT, lat REAL, lon REAL
        )
    """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS person_access (
            name TEXT, surname TEXT, birth_year INT, access_level INT
        )
    """
    )
    db.commit()

    def get_power_plants() -> pl.DataFrame:
        """Get all power plant locations as a DataFrame with city, lat, lon, code columns."""
        res = ai_devs_core.get_power_plants()
        rows = [{"city": city, **attrs} for city, attrs in res["power_plants"].items()]
        df = pl.from_dicts(rows)
        # debugging output into sqlite for visualisation
        db.execute("DELETE FROM power_plants")
        for row in rows:
            db.execute(
                "INSERT INTO power_plants (city, lat, lon, code) VALUES (?, ?, ?, ?)",
                (row.get("city"), row.get("lat"), row.get("lon"), row.get("code")),
            )
        db.commit()
        return df

    def get_people() -> pl.DataFrame:
        """Get the list of suspects from the s01e01 lesson output."""
        return ai_devs_core.read_lesson_output("s01e01")

    def get_person_location(name: str, surname: str) -> dict:
        """Get the location history of a person by name and surname.

        name: First name of the person.
        surname: Last name of the person.
        """
        result = ai_devs_core.check_person_location(name=name, surname=surname)
        # debugging output into sqlite for visualisation
        coords_list = (
            result if isinstance(result, list) else result.get("locations", [])
        )
        for entry in coords_list:
            if isinstance(entry, str):
                parts = entry.strip("() ").split(",")
                lat, lon = float(parts[0]), float(parts[1])
            elif isinstance(entry, dict):
                lat = float(entry.get("lat") or entry.get("latitude") or entry.get("y"))
                lon = float(
                    entry.get("lon") or entry.get("longitude") or entry.get("x")
                )
            else:
                lat, lon = float(entry[0]), float(entry[1])
            db.execute(
                "INSERT INTO person_locations (name, surname, lat, lon) VALUES (?, ?, ?, ?)",
                (name, surname, lat, lon),
            )
        db.commit()
        return result

    # Known accurate coordinates for Polish cities with power plants.
    # Using hardcoded values because LLM geocoding has been unreliable (Chelmno
    # was off by ~120 km causing a missed match).
    _KNOWN_COORDS: dict[str, tuple[float, float]] = {
        "Zabrze": (50.3045, 18.7714),
        "Piotrków Trybunalski": (51.4024, 19.7040),
        "Grudziądz": (53.4838, 18.7538),
        "Tczew": (54.0917, 18.7789),
        "Radom": (51.4027, 21.1470),
        "Chelmno": (53.3488, 18.4296),
        "Żarnowiec": (54.6306, 18.0125),
    }

    def get_location_coords(location: str) -> tuple[float, float]:
        """Get location coordinates as (latitude, longitude).

        location: Name of the place e.g. 'Warsaw'
        """
        if location in _KNOWN_COORDS:
            return _KNOWN_COORDS[location]
        # Fallback to LLM for unknown locations
        client = FAgent("mistral-small-latest")
        prompt = f"Return the latitude and longitude of {location}."

        class Response(pydantic.BaseModel):
            lat: float
            lon: float

        response = client.chat_completion(message=prompt, response_schema=Response)
        return (
            response.choices[0].message.parsed.lat,
            response.choices[0].message.parsed.lon,
        )

    def is_location_close(
        coords1: tuple[float, float], coords2: tuple[float, float]
    ) -> bool:
        """Decide whether the given coordinates are geographically close to each other.

        coords1: float - tuple of x,y coords
        coords2: float - tuple of x,y coords
        """

        def _parse(coords):
            if isinstance(coords, str):
                parts = coords.strip("() ").split(",")
                return (float(parts[0]), float(parts[1]))
            if isinstance(coords[0], str):
                return (float(coords[0]), float(coords[1]))
            return coords

        coords1 = _parse(coords1)
        coords2 = _parse(coords2)

        lat1, lon1 = math.radians(coords1[0]), math.radians(coords1[1])  # radians
        lat2, lon2 = math.radians(coords2[0]), math.radians(coords2[1])  # radians

        dlat = lat2 - lat1  # radians
        dlon = lon2 - lon1  # radians

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))  # radians

        R = 6371.0  # km, mean Earth radius
        distance = R * c  # km

        threshold = 5.0  # km
        return distance <= threshold

    def check_person_access(name: str, surname: str, birth_year: int) -> dict:
        """Check the access level of a person.

        name: First name of the person.
        surname: Last name of the person.
        birth_year: Year of birth of the person.
        """
        result = ai_devs_core.check_person_access(
            name=name, surname=surname, birthYear=birth_year
        )
        # debugging output into sqlite for visualisation
        access_level = result.get("accessLevel") if isinstance(result, dict) else None
        db.execute(
            "INSERT INTO person_access (name, surname, birth_year, access_level) VALUES (?, ?, ?, ?)",
            (name, surname, birth_year, access_level),
        )
        db.commit()
        return result

    class Answer(pydantic.BaseModel):
        name: str
        surname: str
        accessLevel: int
        powerPlant: str

    # Resolve power plant coordinates
    power_plants_df = get_power_plants()
    plant_coords: dict[str, tuple[float, float, str]] = {}
    for plant_row in power_plants_df.iter_rows(named=True):
        city = plant_row["city"]
        lat, lon = get_location_coords(city)
        plant_coords[city] = (lat, lon, plant_row["code"])
        logger.info(f"Power plant {city}: ({lat}, {lon}) code={plant_row['code']}")

    # Check ALL suspects against ALL power plants; pick the closest match
    best: tuple[float, dict, str, str] | None = None  # (dist, person, city, code)
    people_df = get_people()
    for person in people_df.iter_rows(named=True):
        name, surname, born = person["name"], person["surname"], person["born"]
        locations = get_person_location(name, surname)
        coords_list = (
            locations if isinstance(locations, list) else locations.get("locations", [])
        )

        for entry in coords_list:
            if isinstance(entry, str):
                parts = entry.strip("() ").split(",")
                ploc = (float(parts[0]), float(parts[1]))
            elif isinstance(entry, dict):
                ploc = (
                    float(entry.get("lat") or entry.get("latitude") or entry.get("y")),
                    float(entry.get("lon") or entry.get("longitude") or entry.get("x")),
                )
            else:
                ploc = (float(entry[0]), float(entry[1]))

            for city, (clat, clon, code) in plant_coords.items():
                if is_location_close(ploc, (clat, clon)):
                    dist = math.dist(ploc, (clat, clon))  # rough proxy for sorting
                    logger.info(
                        f"Candidate: {name} {surname} near {city} ({code}) ~{dist:.4f}deg"
                    )
                    if best is None or dist < best[0]:
                        best = (dist, person, city, code)

    if best is None:
        logger.error("No match found!")
        db.close()
        return

    _, person, city, code = best
    name, surname, born = person["name"], person["surname"], person["born"]
    logger.info(f"Best match: {name} {surname} near {city} ({code})")
    access = check_person_access(name, surname, born)
    output = Answer(
        name=name,
        surname=surname,
        accessLevel=access.get("accessLevel"),
        powerPlant=code,
    )

    logger.info(output)
    ai_devs_core.verify(task=TASK_NAME, data=dict(output))
    db.close()
    return


if __name__ == "__main__":
    main()
