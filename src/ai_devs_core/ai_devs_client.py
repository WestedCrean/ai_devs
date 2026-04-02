"""
AIDevsClient for interacting with the AIDevs hub API.
"""

import httpx
import pathlib
from loguru import logger
from typing import List, Dict, Any
import polars as pl
from pydantic import json


class AIDevsClient:
    """
    A client for interacting with the AIDevs hub API.

    This client provides methods to send data to the verify endpoint
    and download datasets from the AIDevs hub.
    """

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the AIDevsClient with the given API key.

        Args:
            api_key: The API key for authenticating with the AIDevs hub.
        """
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.Client()
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_api_endpoint(self, endpoint: str) -> str:
        return self.client.get(
            f"{self.api_url}/data/{self.api_key}/{endpoint}", headers=self._headers
        )

    def _post_api_endpoint(self, endpoint: str, body: dict) -> str:
        body["apikey"] = self.api_key
        if endpoint != "verify":
            endpoint = "api/{endpoint}"
        return self.client.post(
            f"{self.api_url}/{endpoint}", headers=self._headers, json=body
        )

    def save_lesson_output(self, lesson_code: str, df: pl.DataFrame):
        output_dir = pathlib.Path("./outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            df.write_csv(output_dir / f"{lesson_code}.csv")
        except Exception as e:
            logger.info(f"Exception: {e}. Writing file to {lesson_code}.json")
            df.write_json(output_dir / f"{lesson_code}.json")

    def read_lesson_output(self, lesson_code: str) -> pl.DataFrame:
        try:
            return pl.read_csv("./outputs/{lesson_code}.csv")
        except Exception:
            logger.info(f"No csv file found. Reading {lesson_code}.json")
            return pl.read_json(f"./outputs/{lesson_code}.json")

    def verify(self, task: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify the submitted data against the specified task.

        Args:
            task: The name of the task.
            data: List of dictionaries containing the data to send.

        Returns:
            Dictionary containing the API response.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        payload = {"task": task, "answer": data}

        response = self._post_api_endpoint(endpoint="verify", body=payload)

        # Log the response for debugging
        logger.info(f"API Response status: {response.status_code}")
        logger.info(f"API Response content: {response.text}")

        response.raise_for_status()
        return response.json()

    def get_dataset(self, dataset: str, save_path: pathlib.Path) -> pl.DataFrame:
        # Construct full file path
        file_path = save_path / f"{dataset}.csv"

        # check if dataset already was downloaded and if so, just read it
        try:
            return pl.read_csv(file_path)
        except FileNotFoundError:
            pass

        response = self._get_api_endpoint(endpoint=f"{dataset}.csv")

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to {file_path}")
        with open(file_path, "w") as f:
            f.write(response.text)

        return pl.read_csv(file_path)

    def get_power_plants(self) -> pl.DataFrame:
        return self._get_api_endpoint(endpoint="findhim_locations.json")

    def check_person_location(self, name: str, surname: str) -> dict:
        return self._post_api_endpoint(
            endpoint="location", body={"name": name, "surname": surname}
        )

    def check_person_access(self, name: str, surname: str, birthYear: int) -> dict:
        return self._post_api_endpoint(
            endpoint="accesslevel",
            body={"name": name, "surname": surname, "birthYear": birthYear},
        )

    def close(self):
        """Close the HTTP client session."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        self.close()
