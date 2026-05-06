import functools
from loguru import logger
import time


def observe_tool(func):
    @functools.wraps(func)
    def wrapper_observe(*args, **kwargs):
        logger.info(f"Called tool: {func.__name__}")
        time.sleep(2)
        res = func(*args, **kwargs)
        logger.info(f"{func.__name__} -> {res}")
        return res

    return wrapper_observe
