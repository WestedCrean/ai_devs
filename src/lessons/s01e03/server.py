import os
import socket
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ngrok
from loguru import logger
from contextlib import asynccontextmanager
from typing import List, Dict

from src.ai_devs_core import FAgent, OAgent, ORAgent, AIDevsClient, Config, get_config
from .session_manager import SessionManager
from .package_api import PackageAPI
from .models import ChatRequest, ChatResponse

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    listener = await run_ngrok_tunnel(3000)
    await send_api_url_to_hub(listener.url())
    yield
    await listener.close()


app = FastAPI(lifespan=lifespan)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
session_manager = SessionManager()
package_api = PackageAPI()


def complete(message: str, session: list[dict] = []):
    # Initialize FAgent
    agent = FAgent(model_id="mistral-large-latest")
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
        tools=[
            package_api.check_package,
            package_api.redirect_package,
            package_api.check_weather,
            package_api.set_last_package_mentioned,
            package_api.get_last_package_mentioned,
        ],
        max_steps=2,  # Limit tool calling iterations
        # max_reflections=1,
        # reflection_model="mistral-small-latest",
    )

    logger.info(response)

    # Extract final response
    return response.choices[0].message.content


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for operators"""
    logger.info(f"POST /chat: SID:{request.sessionID} MSG:{request.msg}")
    try:
        # Get or create session
        session = session_manager.get_session(request.sessionID)

        # Add user message to session
        session.append({"role": "user", "content": request.msg})

        final_response = complete(request.msg, session)

        # Add assistant response to session
        session.append({"role": "assistant", "content": final_response})

        # Update session
        session_manager.update_session(request.sessionID, session)
        logger.info(f"Response: {final_response}")
        return ChatResponse(msg=final_response)

    except Exception as e:
        logger.error(f"Exception: {e}, ")
        raise e


def run_server():
    """Run the FastAPI server"""
    port = 3000
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
    return port


async def run_ngrok_tunnel(port: int):
    config: Config = get_config()
    return await ngrok.forward(
        port,
        authtoken=config.NGROK_AUTHTOKEN,
    )


async def send_api_url_to_hub(api_url: str) -> any:
    config: Config = get_config()

    ai_devs_core = AIDevsClient(
        api_url=config.AI_DEVS_API_URL, api_key=config.AI_DEVS_API_KEY
    )

    logger.info(f"sending api url to hub...")
    res = ai_devs_core.verify(
        task="proxy",
        data={
            "url": f"{api_url}/chat",
            "sessionID": ai_devs_core.get_session_id(),
        },
    )

    logger.info(f"Response: {res}")


if __name__ == "__main__":
    port = run_server()
