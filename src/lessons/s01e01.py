from loguru import logger
from src.ai_devs_core import AIDevsClient, Config, get_config


def main():
    config: Config = get_config()
    logger.info("Configuration loaded.")
    ai_devs_core = AIDevsClient(api_key=config.AI_DEVS_API_KEY)

    return


if __name__ == "__main__":
    main()
