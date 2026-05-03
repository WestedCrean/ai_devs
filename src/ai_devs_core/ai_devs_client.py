"""
AIDevsClient for interacting with the AIDevs hub API.
"""

from dns.e164 import query

import uuid
import time

import httpx
import pathlib
from loguru import logger
from typing import Any
import polars as pl


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

    def _get_api_endpoint(self, endpoint: str) -> httpx.Response:
        full_endpoint_url = f"{self.api_url}/data/{self.api_key}/{endpoint}"
        logger.info(f"GET {full_endpoint_url}")
        return self.client.get(full_endpoint_url, headers=self._headers, timeout=20)

    def _post_api_endpoint(
        self, endpoint: str, body: dict, query_str: str | None = None
    ) -> httpx.Response:
        body["apikey"] = self.api_key
        if endpoint != "verify":
            endpoint = f"api/{endpoint}"

        if query_str:
            endpoint += f"&{query_str}"

        full_endpoint_url = f"{self.api_url}/{endpoint}"
        logger.info(f"POST {full_endpoint_url} {body}")
        res = self.client.post(
            full_endpoint_url, headers=self._headers, json=body, timeout=20
        )
        return res

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
            return pl.read_csv(f"./outputs/{lesson_code}.csv")
        except Exception:
            logger.info(f"No csv file found. Reading {lesson_code}.json")
            return pl.read_json(f"./outputs/{lesson_code}.json")

    def verify(self, task: str, answer: dict[str, Any]) -> dict[str, Any]:
        """
        Send POST {api}/verify to verify response and get flag
        It will send it like that:
        {
            "apikey": <api_key>,
            "task": <task parameter>,
            "answer": <answer parameter>
        }

        Args:
            task: str - the name of the task from excercice
            answer: dict - whatever you need to send in given task

        Returns:
            {
                "status": status code,
                "response": dictionary made from json response
            }
        """
        payload = {"task": task, "answer": answer}
        response = self._post_api_endpoint(endpoint="verify", body=payload)
        result = {"status": response.status_code, "response": response.json()}
        if response.status_code == 402:
            result["action_required"] = (
                "Budget exhausted. STOP immediately. Do NOT call verify again. "
                "First call verify with answer={'prompt': 'reset'} to renew budget, "
                "then fetch a fresh item list with get_list and restart from item 1."
            )
        elif response.status_code not in (200, 201, 202):
            result["action_required"] = (
                f"Request rejected (HTTP {response.status_code}). "
                "Do not send more verify calls. "
                "Read the response body carefully, revise your prompt, then retry."
            )
        return result

    def fetch_file(self, file_url: str) -> str:
        """
        Fetch file content from a given URL.

        Args:
            file_url: str - the URL of the file to fetch
        Returns:
            str - the content of the file
        """
        response = self.client.get(file_url, headers=self._headers, timeout=20)
        response.raise_for_status()
        return response.text

    def _download_dataset(
        self, dataset: str, save_path: pathlib.Path, download_always=False
    ) -> pathlib.Path:
        """
        Download dataset .csv file from api
        """
        file_path = save_path / f"{dataset}.csv"

        response: httpx.Response = self._get_api_endpoint(endpoint=f"{dataset}.csv")

        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to {file_path}")
        with open(file_path, "w") as f:
            f.write(response.text)

        return file_path

    def download_dataset_file(
        self,
        dataset: str,
        save_path: pathlib.Path,
        download_always: bool = False,
    ) -> pathlib.Path:
        """Download a dataset CSV file and return its local path.

        Args:
            dataset: Dataset name without the ``.csv`` suffix.
            save_path: Directory where the downloaded file should be saved.
            download_always: Whether to force downloading a fresh copy.

        Returns:
            Path to the saved CSV file.
        """
        return self._download_dataset(
            dataset=dataset,
            save_path=save_path,
            download_always=download_always,
        )

    def get_dataset(
        self,
        dataset: str,
        save_path: pathlib.Path,
        mode="dataframe",
        download_always=False,
    ) -> pl.DataFrame:
        """
        Download dataset .csv file from api and read it in appropriate mode
        """

        file_path = self._download_dataset(
            dataset=dataset, save_path=save_path, download_always=download_always
        )

        if mode == "string":
            with open(file_path, "r") as f:
                return pl.DataFrame({"line": f.readlines()})

        return pl.read_csv(file_path)

    def get_dataset_as_lines(
        self,
        dataset: str,
        save_path: pathlib.Path,
        download_always=False,
    ) -> list[str]:
        """
        Download dataset .csv file from api and read it in appropriate mode
        """

        file_path = self._download_dataset(
            dataset=dataset, save_path=save_path, download_always=download_always
        )

        with open(file_path, "r") as f:
            return f.readlines()

    def get_dataset_as_dataframe(
        self,
        dataset: str,
        save_path: pathlib.Path,
        download_always=False,
    ) -> pl.DataFrame:
        """
        Download dataset .csv file from api and read it as polars dataframe
        """
        file_path = self._download_dataset(
            dataset=dataset, save_path=save_path, download_always=download_always
        )

        return pl.read_csv(file_path)

    def get_power_plants(self) -> dict[str, Any]:
        """
        Get data from findhim_locations.json
        """
        return self._get_api_endpoint(endpoint="findhim_locations.json").json()

    def check_person_location(self, name: str, surname: str) -> httpx.Response:
        return self._post_api_endpoint(
            endpoint="location", body={"name": name, "surname": surname}
        )

    def check_person_access(
        self, name: str, surname: str, birthYear: int
    ) -> httpx.Response:
        """
        Check person access using /accesslevel endpoint
        """
        return self._post_api_endpoint(
            endpoint="accesslevel",
            body={"name": name, "surname": surname, "birthYear": birthYear},
        )

    def sleep(seconds: int = 10) -> None:
        """
        Tool for waiting.

        seconds: int - how many seconds to wait (default 10)
        """
        time.sleep(seconds)

    def get_session_id(self) -> str:
        return str(uuid.uuid4())

    def close(self):
        """Close the HTTP client session."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        self.close()
