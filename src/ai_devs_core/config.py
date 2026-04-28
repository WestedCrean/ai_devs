import os
from typing import Optional, cast
from dotenv import dotenv_values
from pydantic import BaseModel


class Config(BaseModel):
    AI_DEVS_API_URL: str
    AI_DEVS_API_KEY: str
    MISTRAL_API_KEY: str
    OPENROUTER_API_KEY: str
    OPENAI_API_KEY: str
    OPENAI_API_BASE_URL: str = "https://openrouter.ai/api/v1"
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str
    WANDB_API_KEY: str
    LM_STUDIO_KEY: str
    LM_STUDIO_URL: str
    NGROK_AUTHTOKEN: str

    model_to_use: str = "mistral-small-2603"


def get_config() -> Config:
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    config = dotenv_values(env_path)
    merged = {**config, **os.environ}
    normalized = {key: cast(str, value or "") for key, value in merged.items()}
    return Config(**normalized)


class BatchJobConfig(BaseModel):
    """
    Configuration for batch job processing.

    Attributes:
        model: Mistral model to use for processing
        poll_interval: Seconds between job status polls
        timeout: Maximum time to wait for job completion (seconds)
        max_workers: Maximum number of parallel workers for fallback processing
        max_retries: Maximum number of retries for failed requests
        chunk_size: Size of chunks for processing large datasets
        fallback_model: Model to use when batch API fails
        retry_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        rate_limit: Maximum requests per second (0 = unlimited)
        correlation_id: Optional correlation ID for tracing
    """

    model: str = "mistral-small-latest"
    poll_interval: int = 10
    timeout: int = 120
    max_workers: int = 1
    max_retries: int = 5
    chunk_size: int = 1000
    fallback_model: str = "mistral-medium-latest"
    retry_delay: float = 1.0
    max_delay: float = 60.0
    rate_limit: int = 0
    correlation_id: Optional[str] = None
