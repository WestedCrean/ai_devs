import os
from typing import Dict
from dotenv import dotenv_values
from pydantic import BaseModel


class Config(BaseModel):
    AI_DEVS_API_KEY: str
    MISTRAL_API_KEY: str
    OPENROUTER_API_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_BASE_URL: str
    WANDB_API_KEY: str


def get_config() -> Config:
    env_path = os.path.join(os.path.dirname(__file__), "../../.env")
    config = dotenv_values(env_path)
    return Config(**config)
