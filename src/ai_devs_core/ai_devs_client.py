"""
AIDevsClient for interacting with the AIDevs hub API.
"""

import httpx
import pathlib
from loguru import logger
from typing import List, Dict, Any
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
        payload = {"apikey": self.api_key, "task": task, "answer": data}

        response = self.client.post(
            f"{self.api_url}/verify", json=payload, headers=self._headers
        )

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

        response = self.client.get(
            f"{self.api_url}/data/{self.api_key}/{dataset}.csv", headers=self._headers
        )
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to {file_path}")
        with open(file_path, "w") as f:
            f.write(response.text)

        return pl.read_csv(file_path)

    def close(self):
        """Close the HTTP client session."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        self.close()
