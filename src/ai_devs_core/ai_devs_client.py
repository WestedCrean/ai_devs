"""
AIDevsClient for interacting with the AIDevs hub API.
"""

import httpx
from typing import List, Dict, Any
import polars as pl


class AIDevsClient:
    """
    A client for interacting with the AIDevs hub API.

    This client provides methods to send data to the verify endpoint
    and download datasets from the AIDevs hub.
    """

    BASE_URL = "https://hub.ag3nts.org"

    def __init__(self, api_key: str):
        """
        Initialize the AIDevsClient with the given API key.

        Args:
            api_key: The API key for authenticating with the AIDevs hub.
        """
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
            f"{self.BASE_URL}/verify", json=payload, headers=self._headers
        )

        response.raise_for_status()
        return response.json()

    def download_dataset(self, dataset: str, save_path: str) -> pl.DataFrame:
        # check if dataset already was downloaded and if so, just read it
        try:
            return pl.read_csv(save_path)
        except FileNotFoundError:
            pass

        response = self.client.get(
            f"{self.BASE_URL}/data/{self.api_key}/{dataset}", headers=self._headers
        )
        print(f"Saving to {save_path}")
        with open(save_path, "w") as f:
            f.write(response.text)

        return pl.read_csv(save_path)

    def close(self):
        """Close the HTTP client session."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        self.close()
