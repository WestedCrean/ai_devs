from pathlib import Path
from dotenv import load_dotenv

from pydantic import BaseModel

from loguru import logger
import uvicorn
import ngrok
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


from src.ai_devs_core import FAgent, OAgent, ORAgent, AIDevsClient, Config, get_config

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # listener = await run_ngrok_tunnel(3000)
    # await send_api_url_to_hub(listener.url())
    yield
    # await listener.close()


app = FastAPI(lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """

"""

mcp = {}


def complete(
    message: str, session: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
):
    agent = FAgent(model_id="mistral-small-latest")
    # agent = OAgent(
    #     model_id="gemma-4-e4b-uncensored-hauhaucs-aggressive",
    #     api_base=config.LM_STUDIO_URL,
    #     api_key=config.LM_STUDIO_KEY,
    # )
    # agent = ORAgent(model_id="moonshotai/kimi-k2-thinking")
    # Process with FAgent
    response = agent.chat_completion(
        message=message,
        chat_history=session,
        tools=[],
        mcp_definition=mcp,
        max_steps=2,  # Limit tool calling iterations
        # max_reflections=1,
        # reflection_model="mistral-small-latest",
    )
    logger.info(response)
    return response.choices[0].message.content


async def main(query):
    """Main chat endpoint for operators"""
    try:
        final_response = complete(message=query)
        logger.info(f"Response: {final_response}")
    except Exception as e:
        logger.error(f"Exception: {e}, ")
        raise e


if __name__ == "__main__":
    query = ""
    main(query=query)
